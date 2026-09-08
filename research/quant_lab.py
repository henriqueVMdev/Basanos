# -*- coding: utf-8 -*-
"""Laboratorio de estrategias 15m com EXECUCAO HONESTA (prop challenge).

Regras de execucao (a prova de simulador otimista):
  - Sinal apenas com candle FECHADO; entrada A MERCADO no open do candle
    seguinte (taker + slippage). Nenhuma hipotese de fila maker na entrada.
  - Stop protetivo intrabar: sai no preco do stop com taker + slippage.
    Stop e alvo no mesmo candle = STOP (conservador).
  - Alvo (quando usado): limite reduce-only; fill exige ATRAVESSAR o nivel
    (estrito) e respeita a forma da barra no candle da entrada.
  - Saida por tempo/condicao: a mercado no CLOSE do candle (taker + slip).
  - Funding real por posicao: adverso x1.5, favoravel x0.5 (stress).
  - Custos Bybit: maker 0.02% / taker 0.055% / slip 0.05% por ponta taker.

Metodologia: 12 perps Bybit, 1 ano 15m, split 70/30. Este script roda as
familias candidatas APENAS NO TREINO. O forward so e tocado uma vez, por
research/quant_forward.py, depois da selecao.

Uso: py research/quant_lab.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "data")
MAKER, TAKER, SLIP = 0.0002, 0.00055, 0.0005
SYMS = ["btc", "eth", "sol", "xrp", "doge", "bnb", "ada", "link",
        "avax", "sui", "ltc", "near"]


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def atr(h, l, c, n=14):
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    tr = pd.concat([hs - ls, (hs - cs.shift()).abs(),
                    (ls - cs.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


def run_engine(df, fund, signal, cfg):
    """Motor honesto. signal: array com +1/-1/0 avaliado no candle FECHADO i
    (entrada no open de i+1). cfg: dict com stop_atr, tp_atr (0=sem alvo),
    max_bars, trail (bool)."""
    O, H, L, C = (df[c].values for c in ["Open", "High", "Low", "Close"])
    TS = ((df.index - pd.Timestamp(0)).total_seconds().values * 1000).astype(np.int64)
    A = atr(H, L, C)
    N = len(df)
    fts, fr = fund["timestamp"].values, fund["rate"].values
    stop_atr = cfg.get("stop_atr", 2.0)
    tp_atr = cfg.get("tp_atr", 0.0)
    max_bars = cfg.get("max_bars", 96)
    trail = cfg.get("trail", False)

    trades = []
    i = 1
    while i < N - 1:
        s = signal[i]
        if s == 0 or np.isnan(A[i]) or A[i] <= 0:
            i += 1
            continue
        side = int(s)
        j = i + 1                       # candle da entrada
        entry = O[j] * (1 + SLIP * side)          # taker + slip na entrada
        stop = entry - side * stop_atr * A[i]
        tp = entry + side * tp_atr * A[i] if tp_atr > 0 else None

        exit_price = None
        reason = ""
        k = j
        while k < N:
            if trail and k > j:
                new_stop = C[k - 1] - side * stop_atr * A[k - 1]
                stop = max(stop, new_stop) if side == 1 else min(stop, new_stop)
            hit_sl = (L[k] <= stop) if side == 1 else (H[k] >= stop)
            hit_tp = False
            if tp is not None:
                hit_tp = (H[k] > tp) if side == 1 else (L[k] < tp)
                if k == j and hit_tp:
                    # forma da barra: alvo precisa vir DEPOIS da abertura
                    hit_tp = (C[k] >= O[k]) if side == 1 else (C[k] < O[k])
            if hit_sl:                  # stop primeiro: mesmo candle = LOSS
                exit_price = stop * (1 - SLIP * side)
                reason = "stop"
                break
            if hit_tp:
                exit_price = tp         # limite maker (cruzou o nivel)
                reason = "alvo"
                break
            if k - j + 1 >= max_bars:
                exit_price = C[k] * (1 - SLIP * side)
                reason = "tempo"
                break
            k += 1
        if exit_price is None:
            k = N - 1
            exit_price = C[k] * (1 - SLIP * side)
            reason = "fim"

        gross = side * (exit_price / entry - 1) * 100
        fee = (TAKER + (MAKER if reason == "alvo" else TAKER)) * 100
        i0 = np.searchsorted(fts, TS[j], side="right")
        i1 = np.searchsorted(fts, TS[k], side="right")
        f = fr[i0:i1].sum() if i1 > i0 else 0.0
        f_signed = (f if side == 1 else -f) * 100      # >0 = paga
        f_cost = f_signed * 1.5 if f_signed > 0 else f_signed * 0.5
        net = gross - fee - f_cost
        stop_dist_pct = stop_atr * A[i] / entry * 100
        trades.append((net, TS[j], TS[k], side, stop_dist_pct))
        i = k + 1
    return trades


# Familias de sinal (todas no candle fechado i)
def sig_donchian(df, n=96, trend=200):
    """Rompimento do maior high/menor low de n candles, filtro EMA de tendencia."""
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    hh = pd.Series(H).rolling(n).max().shift(1).values
    ll = pd.Series(L).rolling(n).min().shift(1).values
    e = ema(C, trend)
    sig = np.zeros(len(df))
    sig[(C > hh) & (C > e)] = 1
    sig[(C < ll) & (C < e)] = -1
    return sig


def sig_momentum(df, n=96, thr=1.0):
    """Momentum de n candles: |retorno| acima de thr%% entra a favor."""
    C = df["Close"].values
    ret = pd.Series(C).pct_change(n).values * 100
    sig = np.zeros(len(df))
    sig[ret > thr] = 1
    sig[ret < -thr] = -1
    return sig


def sig_session_break(df, start=13, end=20):
    """Rompimento do range da madrugada (00-13h UTC) na sessao americana."""
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    hour = df.index.hour.values
    day = df.index.date
    sig = np.zeros(len(df))
    cur_day = None
    hi = lo = None
    for i in range(len(df)):
        if day[i] != cur_day:
            cur_day = day[i]
            hi = lo = None
        if hour[i] < start:
            hi = H[i] if hi is None else max(hi, H[i])
            lo = L[i] if lo is None else min(lo, L[i])
        elif hour[i] < end and hi is not None:
            if C[i] > hi:
                sig[i] = 1
                hi = np.inf            # um rompimento por dia
            elif C[i] < lo:
                sig[i] = -1
                lo = -np.inf
    return sig


def sig_flush_reversal(df, n=16, z=3.0):
    """Reversao apos flush: retorno de n candles alem de z desvios."""
    C = df["Close"].values
    ret = pd.Series(C).pct_change(n)
    mu = ret.rolling(672).mean()
    sd = ret.rolling(672).std()
    zs = ((ret - mu) / sd).values
    sig = np.zeros(len(df))
    sig[zs < -z] = 1
    sig[zs > z] = -1
    return sig


def sig_funding_mom(df, fund, n=96, thr=0.5, f_thr=0.0):
    """Momentum filtrado por funding: so entra no lado que NAO paga funding."""
    base = sig_momentum(df, n, thr)
    TS = ((df.index - pd.Timestamp(0)).total_seconds().values * 1000).astype(np.int64)
    fts, fr = fund["timestamp"].values, fund["rate"].values
    idx = np.searchsorted(fts, TS, side="right") - 1
    idx = np.clip(idx, 0, len(fr) - 1)
    f_now = fr[idx]
    sig = base.copy()
    sig[(base == 1) & (f_now > f_thr)] = 0     # long pagaria funding alto
    sig[(base == -1) & (f_now < -f_thr)] = 0   # short pagaria funding alto
    return sig


FAMILIES = {
    "donchian 24h s2 t8":   (lambda d, f: sig_donchian(d, 96),  dict(stop_atr=2, max_bars=32, trail=False)),
    "donchian 24h trail3":  (lambda d, f: sig_donchian(d, 96),  dict(stop_atr=3, max_bars=192, trail=True)),
    "donchian 48h trail3":  (lambda d, f: sig_donchian(d, 192), dict(stop_atr=3, max_bars=192, trail=True)),
    "mom 24h 1.5% s2 t24h": (lambda d, f: sig_momentum(d, 96, 1.5), dict(stop_atr=2, max_bars=96)),
    "mom 24h 2.5% trail3":  (lambda d, f: sig_momentum(d, 96, 2.5), dict(stop_atr=3, max_bars=192, trail=True)),
    "sessao americana s2":  (lambda d, f: sig_session_break(d), dict(stop_atr=2, max_bars=44)),
    "flush z3 tp2 s2":      (lambda d, f: sig_flush_reversal(d), dict(stop_atr=2, tp_atr=2, max_bars=32)),
    "flush z3 t8h":         (lambda d, f: sig_flush_reversal(d), dict(stop_atr=2.5, max_bars=32)),
    "fundmom 24h1.5 s2":    (lambda d, f: sig_funding_mom(d, f, 96, 1.5), dict(stop_atr=2, max_bars=96)),
    "fundmom 24h1.5 trail": (lambda d, f: sig_funding_mom(d, f, 96, 1.5), dict(stop_atr=3, max_bars=192, trail=True)),
}


def stats(tr):
    if not tr:
        return None
    p = np.array([t[0] for t in tr])
    loss = abs(p[p <= 0].sum())
    pf = p[p > 0].sum() / loss if loss > 0 else float("inf")
    return dict(n=len(p), wr=(p > 0).mean() * 100, exp=p.mean(), pf=pf)


if __name__ == "__main__":
    data = {}
    for sym in SYMS:
        try:
            df = pd.read_csv(os.path.join(SCRATCH, f"{sym}_15m_bybit.csv"),
                             index_col=0, parse_dates=True)
            fund = pd.read_csv(os.path.join(SCRATCH, f"funding_{sym}.csv"))
        except FileNotFoundError:
            continue
        cut = int(len(df) * 0.7)
        data[sym] = (df.iloc[:cut], fund)      # SO TREINO

    print(f"{len(data)} simbolos | apenas TREINO (70%)")
    print(f"{'familia':22s} {'n':>6s} {'WR':>6s} {'exp/trade':>10s} {'PF':>6s} {'pos':>5s} {'tr/d':>5s}")
    for name, (sigfn, cfg) in FAMILIES.items():
        allt = []
        pos = 0
        for sym, (df, fund) in data.items():
            tr = run_engine(df, fund, sigfn(df, fund), cfg)
            if tr:
                p = np.array([t[0] for t in tr])
                if p.mean() > 0:
                    pos += 1
                allt += tr
        s = stats(allt)
        if not s:
            print(f"{name:22s}   sem trades")
            continue
        days = 255 * len(data)
        print(f"{name:22s} {s['n']:6d} {s['wr']:5.1f}% {s['exp']:+10.4f} "
              f"{s['pf']:6.2f} {pos:3d}/12 {s['n']/days:5.2f}")
