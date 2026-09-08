# -*- coding: utf-8 -*-
"""Otimizacao ANTI-overfit do nucleo da DePaula (angulo de MA + histerese)
no DIARIO, 13 perps, com o motor honesto do research (run_daily: sinal no
fechamento -> entrada na abertura seguinte, taker+slip+funding).

Por que nao usar backtesting.py: executa no close da mesma barra do sinal,
sem custos, e checa TP/SL intrabar contra bandas da MA da propria barra
(lookahead). Aqui so o SINAL da DePaula e mantido; a execucao e a honesta.
Fidelidade ao original: MA (SMA/EMA/HMA), slope por OLS no lookback,
angulo = atan(slope/ATR_rma) em graus, maquina de estados com histerese.
Unica correcao: sem o clamp max(atr, 0.01) do mintick (distorcia moedas
de preco baixo); usa atr>0.

Protocolo: grid 100%% no TREINO (<2024); plato nos vizinhos; validacao
2024-25 UMA vez com pacote pre-comprometido (--val); 2026 intocado.

Uso: py research/depaula_opt.py [--val]
"""

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daily_lab import SCRATCH, SYMS, run_daily, TRAIN_END

VAL_END = "2026-01-01"
HYST = 0.2

data = {}
for sym in SYMS:
    try:
        data[sym] = pd.read_csv(os.path.join(SCRATCH, f"{sym}_1d_binance.csv"),
                                index_col=0, parse_dates=True)
    except FileNotFoundError:
        continue


# indicadores (mesma matematica do backtesting.py)
def _wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    return pd.Series(x).rolling(n).apply(
        lambda v: np.dot(v, w) / w.sum(), raw=True).values


def calc_ma(x, n, ma_type):
    if ma_type == "SMA":
        return pd.Series(x).rolling(n).mean().values
    if ma_type == "EMA":
        return pd.Series(x).ewm(span=n, adjust=False).mean().values
    if ma_type == "HMA":
        half, sq = max(n // 2, 1), max(int(math.sqrt(n)), 1)
        return _wma(2 * _wma(x, half) - _wma(x, n), sq)
    raise ValueError(ma_type)


def atr_rma(df, n=14):
    H, L, C = df["High"], df["Low"], df["Close"]
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean().values


def slope_ols(ma, n):
    """Slope OLS sobre janela n (x = -(n-1)..0) — vetorizada por convolucao."""
    n = max(n, 2)
    xw = np.arange(-(n - 1), 1, dtype=float)
    sx, sxx = xw.sum(), (xw * xw).sum()
    y = np.nan_to_num(ma, nan=0.0)
    sy = np.convolve(y, np.ones(n), "full")[n - 1:len(ma)]
    sxy = np.convolve(y, xw[::-1], "full")[n - 1:len(ma)]
    out = np.zeros(len(ma))
    out[n - 1:] = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    # janela com NaN na MA -> slope 0 (mesmo comportamento do original)
    bad = pd.Series(np.isnan(ma)).rolling(n).max().fillna(1).values > 0
    out[bad] = 0.0
    return out


def angle_series(df, ma_type, ma_len, lookback):
    ma = calc_ma(df["Close"].values, ma_len, ma_type)
    a = atr_rma(df)
    sl = slope_ols(ma, lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(a > 0, sl / a, sl)
    ang = np.degrees(np.arctan(norm))
    ang[np.isnan(ma) | np.isnan(a)] = 0.0
    return ang


def states_hyst(ang, th):
    """Maquina de estados do original: th_up=th, th_dn=-th, histerese HYST."""
    s = np.zeros(len(ang))
    st = 0
    for i in range(len(ang)):
        a = ang[i]
        if st <= 0 and a > th + HYST:
            st = 1
        elif st >= 0 and a < -th - HYST:
            st = -1
        elif st == 1 and a < th - HYST:
            st = 0
        elif st == -1 and a > -th + HYST:
            st = 0
        s[i] = st
    return s


def variant_signal(states, mode, long_only):
    s = states
    if mode == "flip":          # exit_on_flat=False: segura ate inverter
        s = pd.Series(s).replace(0, np.nan).ffill().fillna(0).values
    if long_only:
        s = np.where(s > 0, 1, 0).astype(float)
    return s


# grid no treino
MA_TYPES = ["SMA", "EMA", "HMA"]
MA_LENS = [20, 50, 100, 200]
LOOKBACKS = [3, 5, 10]
THS = [0.3, 0.5, 1.0, 2.0]
VARIANTS = [("flat", False), ("flat", True), ("flip", False), ("flip", True)]


def portfolio_row(frames):
    allt = pd.concat(frames)
    p = allt["net_pct"].values
    if len(p) < 150:
        return None
    loss = abs(p[p <= 0].sum())
    pf = p[p > 0].sum() / loss if loss > 0 else float("inf")
    days = (allt["ts_exit"].max() - allt["ts_entry"].min()) / 86400000
    tpy = len(p) / max(days / 365.25, 0.1)
    sh = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0
    anos = pd.to_datetime(allt["ts_entry"], unit="ms").dt.year.values
    yr_sum = {a: p[anos == a].sum() for a in range(2020, 2024)}
    return dict(n=len(p), wr=(p > 0).mean() * 100, exp=p.mean(), pf=pf,
                sharpe=sh, yrs_pos=sum(v > 0 for v in yr_sum.values()),
                **{f"y{a}": v for a, v in yr_sum.items()})


def collect(sigs, t0=None, t1=None):
    """sigs: dict {sym: array de sinal} ja calculado."""
    frames = []
    for sym, df in data.items():
        tr = run_daily(df, sigs[sym], dict())
        if len(tr):
            if t0:
                tr = tr[tr["ts_entry"] >= pd.Timestamp(t0).value // 10**6]
            if t1:
                tr = tr[tr["ts_entry"] < pd.Timestamp(t1).value // 10**6]
            tr["symbol"] = sym
            frames.append(tr)
    return frames


if __name__ == "__main__":
    if "--val" in sys.argv:
        # PACOTE PRE-COMPROMETIDO (escolhido no treino 2026-07-09):
        # centro do plato "todos os anos positivos" — EMA 100, lb 5,
        # th 0.5, exit_on_flat, L/S (vizinhos 50/200, lb 3-10, th 0.3-1.0,
        # flat/flip todos com PF 7+ e 2020-23 positivos)
        PACKAGE = lambda df: variant_signal(
            states_hyst(angle_series(df, "EMA", 100, 5), 0.5), "flat", False)
        if PACKAGE is None:
            print("edite PACKAGE com o pacote escolhido no treino")
            sys.exit(1)
        frames = collect({sym: PACKAGE(df) for sym, df in data.items()},
                         TRAIN_END, VAL_END)
        r = portfolio_row(frames)
        print(f"VALIDACAO {TRAIN_END}->{VAL_END}:", r)
        sys.exit(0)

    rows = []
    for mt in MA_TYPES:
        for ml in MA_LENS:
            for lb in LOOKBACKS:
                angs = {sym: angle_series(df, mt, ml, lb)
                        for sym, df in data.items()}
                for th in THS:
                    sts = {sym: states_hyst(angs[sym], th)
                           for sym in data}
                    for mode, lo in VARIANTS:
                        sigs = {sym: variant_signal(sts[sym], mode, lo)
                                for sym in data}
                        frames = collect(sigs, None, TRAIN_END)
                        r = portfolio_row(frames)
                        if r:
                            rows.append(dict(ma=mt, len=ml, lb=lb, th=th,
                                             mode=mode, lo=lo, **r))
                print(f"{mt} {ml} lb={lb} ok", flush=True)

    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(SCRATCH, "depaula_grid.csv"), index=False)
    g = g.sort_values("sharpe", ascending=False)
    cols = ["ma", "len", "lb", "th", "mode", "lo", "n", "wr", "exp", "pf",
            "sharpe", "yrs_pos", "y2020", "y2021", "y2022", "y2023"]
    print(f"\nTREINO <{TRAIN_END} | {len(g)} combos validos | top 30 por sharpe:")
    print(g[cols].head(30).to_string(index=False,
                                     float_format=lambda x: f"{x:+.2f}"))
    print("\ntodos os anos positivos (2020-23), top 15 por PF:")
    gp = g[g["yrs_pos"] == 4].sort_values("pf", ascending=False)
    print(gp[cols].head(15).to_string(index=False,
                                      float_format=lambda x: f"{x:+.2f}"))
