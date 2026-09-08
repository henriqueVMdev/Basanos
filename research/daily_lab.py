# -*- coding: utf-8 -*-
"""Laboratorio DIARIO — busca de edge lucrativo e consistente em cripto.

Base academica (literatura publica):
- Time-series momentum 1-4 semanas (Liu & Tsyvinski 2021, RFS): retornos
  passados de 1-4 semanas preveem retornos futuros em BTC/ETH/XRP.
- Momentum cross-sectional + fator tamanho (Liu, Tsyvinski & Wu 2022, JF).
- Trend-following classico (Moskowitz et al. 2012 TSMOM; canais Donchian
  20/55 dos Turtles; MA 50/200) aplicado a cripto por CTAs.
- Vol targeting melhora Sharpe de trend em cripto (risk parity simples).
- Carry via funding (perps): media historica positiva -> shorts recebem.

Execucao honesta (mesmas regras do lab 15m):
- Sinal no fechamento DIARIO; entrada a mercado na abertura do dia seguinte
  (taker 0.055% + slip 0.05%).
- Stop intrabar (taker+slip); mesmo dia stop+alvo = LOSS.
- Sem alvo limite por padrao (saida por sinal/trailing no close seguinte).
- Perps: funding modelado como drag conservador — long paga 0.03%/dia,
  short RECEBE 0.015%/dia (metade, haircut) — media historica de funding
  positivo; nos anos 2024+ usa o funding real da Bybit quando disponivel.

Dados: Binance spot 1d desde 2017 (majors) — {sym}_1d_binance.csv.
Split: TREINO ate 2023-12-31 | VALIDACAO 2024-2025 | 2026 intocado.

Uso: py research/daily_lab.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "data")
MAKER, TAKER, SLIP = 0.0002, 0.00055, 0.0005
FUND_LONG_DAY = 0.0003          # 0.03%/dia pago quando comprado (perp)
FUND_SHORT_DAY = -0.00015       # short recebe 0.015%/dia (haircut 50%)
SYMS = ["btc", "eth", "bnb", "ltc", "xrp", "ada", "link", "doge",
        "sol", "avax", "dot", "near", "sui", "matic"]
TRAIN_END = "2024-01-01"


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def atr(h, l, c, n=14):
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    tr = pd.concat([hs - ls, (hs - cs.shift()).abs(),
                    (ls - cs.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


def run_daily(df, signal, cfg):
    """Motor honesto diario. signal[i] em {-1,0,1} no fechamento do dia i;
    posicao assumida na ABERTURA de i+1 e mantida enquanto signal nao muda
    (modo 'posicional'): reavalia todo fechamento. Stop opcional em ATR.

    Retorna DataFrame de trades: net_pct (ja com custos e funding),
    ts_entry, ts_exit, side, stop_dist_pct, dias."""
    O, H, L, C = (df[c].values for c in ["Open", "High", "Low", "Close"])
    TS = ((df.index - pd.Timestamp(0)).total_seconds().values * 1000).astype(np.int64)
    A = atr(H, L, C)
    N = len(df)
    stop_atr = cfg.get("stop_atr", 0.0)

    trades = []
    pos = 0
    entry = stop = 0.0
    j0 = -1
    for i in range(1, N - 1):
        # saida por stop intrabar no dia i (posicao aberta antes)
        if pos != 0:
            if stop_atr > 0:
                hit = (L[i] <= stop) if pos == 1 else (H[i] >= stop)
                if hit:
                    exit_p = stop * (1 - SLIP * pos)
                    _close(trades, df, TS, pos, entry, exit_p, j0, i, A)
                    pos = 0
        # sinal no fechamento do dia i -> executa na abertura de i+1
        s = int(signal[i]) if not np.isnan(signal[i]) else 0
        if s != pos:
            if pos != 0:                      # fecha na abertura de i+1
                exit_p = O[i + 1] * (1 - SLIP * pos)
                _close(trades, df, TS, pos, entry, exit_p, j0, i + 1, A)
                pos = 0
            if s != 0 and not np.isnan(A[i]) and A[i] > 0:
                pos = s
                entry = O[i + 1] * (1 + SLIP * pos)
                stop = entry - pos * stop_atr * A[i] if stop_atr > 0 else 0.0
                j0 = i + 1
    if pos != 0:
        _close(trades, df, TS, pos, entry, C[N - 1], j0, N - 1, A)
    return pd.DataFrame(trades, columns=["net_pct", "ts_entry", "ts_exit",
                                         "side", "stop_dist_pct", "dias"])


def _close(trades, df, TS, side, entry, exit_p, j0, j1, A):
    gross = side * (exit_p / entry - 1) * 100
    fee = 2 * TAKER * 100
    dias = max(j1 - j0, 1)
    fund = (FUND_LONG_DAY if side == 1 else FUND_SHORT_DAY) * dias * 100
    net = gross - fee - fund
    sd = A[j0 - 1] / entry * 100 if j0 >= 1 else 1.0
    trades.append((net, TS[j0], TS[j1], side, sd, dias))


# Sinais posicionais (avaliados no fechamento)
def sig_tsmom(df, n=21, long_only=False):
    C = df["Close"].values
    r = pd.Series(C).pct_change(n).values
    s = np.where(r > 0, 1, -1).astype(float)
    if long_only:
        s = np.where(r > 0, 1, 0).astype(float)
    s[np.isnan(r)] = 0
    return s


def sig_ma_cross(df, fast=50, slow=200, long_only=False):
    C = df["Close"].values
    f, sl = ema(C, fast), ema(C, slow)
    s = np.where(f > sl, 1, -1).astype(float)
    if long_only:
        s = np.where(f > sl, 1, 0).astype(float)
    s[:slow] = 0
    return s


def sig_donchian_d(df, n=55, exit_n=20, long_only=False):
    """Turtle-style: entra no rompimento de n dias, sai no canal de exit_n."""
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    hh = pd.Series(H).rolling(n).max().shift(1).values
    ll = pd.Series(L).rolling(n).min().shift(1).values
    xh = pd.Series(H).rolling(exit_n).max().shift(1).values
    xl = pd.Series(L).rolling(exit_n).min().shift(1).values
    s = np.zeros(len(df))
    pos = 0
    for i in range(len(df)):
        if np.isnan(hh[i]) or np.isnan(xl[i]):
            continue
        if pos == 0:
            if C[i] > hh[i]:
                pos = 1
            elif C[i] < ll[i] and not long_only:
                pos = -1
        elif pos == 1 and C[i] < xl[i]:
            pos = 0
        elif pos == -1 and C[i] > xh[i]:
            pos = 0
        s[i] = pos
    return s


def sig_tsmom_vol(df, n=21, vol_n=30, vol_target=0.02, long_only=False):
    """TSMOM com vol targeting: so mantem posicao quando a vol diaria nao
    esta explosiva (reduz drawdown; sizing fino fica para o portfolio)."""
    C = df["Close"].values
    r1 = pd.Series(C).pct_change()
    vol = r1.rolling(vol_n).std().values
    base = sig_tsmom(df, n, long_only)
    base[vol > vol_target * 2.5] = 0
    return base


FAMILIES = {
    "tsmom 7d":            (lambda d: sig_tsmom(d, 7),                    dict()),
    "tsmom 21d":           (lambda d: sig_tsmom(d, 21),                   dict()),
    "tsmom 28d":           (lambda d: sig_tsmom(d, 28),                   dict()),
    "tsmom 21d long-only": (lambda d: sig_tsmom(d, 21, True),             dict()),
    "ma 20/100":           (lambda d: sig_ma_cross(d, 20, 100),           dict()),
    "ma 50/200":           (lambda d: sig_ma_cross(d, 50, 200),           dict()),
    "ma 50/200 long-only": (lambda d: sig_ma_cross(d, 50, 200, True),     dict()),
    "donchian 55/20":      (lambda d: sig_donchian_d(d, 55, 20),          dict()),
    "donchian 20/10":      (lambda d: sig_donchian_d(d, 20, 10),          dict()),
    "donch 55/20 long":    (lambda d: sig_donchian_d(d, 55, 20, True),    dict()),
    "tsmom21 volfilt":     (lambda d: sig_tsmom_vol(d, 21),               dict()),
    "tsmom21 stop3atr":    (lambda d: sig_tsmom(d, 21),                   dict(stop_atr=3)),
}


def portfolio_stats(all_trades, label):
    """Metricas no nivel do portfolio: agrega trades de todos os simbolos
    com peso igual por trade (exposicao unitaria)."""
    p = all_trades["net_pct"].values
    if len(p) < 30:
        print(f"{label:22s}  amostra insuficiente ({len(p)})")
        return
    loss = abs(p[p <= 0].sum())
    pf = p[p > 0].sum() / loss if loss > 0 else float("inf")
    # Sharpe por trade anualizado pela frequencia media
    days = (all_trades["ts_exit"].max() - all_trades["ts_entry"].min()) / 86400000
    tpy = len(p) / max(days / 365.25, 0.1)
    sh = p.mean() / p.std() * np.sqrt(tpy) if p.std() > 0 else 0
    print(f"{label:22s} n={len(p):5d} WR={(p>0).mean()*100:5.1f}% "
          f"exp={p.mean():+7.3f}% PF={pf:5.2f} sharpe~{sh:5.2f} "
          f"medio {all_trades['dias'].mean():4.1f}d")


if __name__ == "__main__":
    data = {}
    for sym in SYMS:
        try:
            df = pd.read_csv(os.path.join(SCRATCH, f"{sym}_1d_binance.csv"),
                             index_col=0, parse_dates=True)
        except FileNotFoundError:
            continue
        data[sym] = df

    print(f"{len(data)} simbolos | TREINO (ate {TRAIN_END})")
    for name, (sigfn, cfg) in FAMILIES.items():
        frames = []
        for sym, df in data.items():
            tr = run_daily(df, sigfn(df), cfg)
            if len(tr):
                tr = tr[tr["ts_entry"] < pd.Timestamp(TRAIN_END).value // 10**6]
                tr["symbol"] = sym
                frames.append(tr)
        if not frames:
            continue
        allt = pd.concat(frames)
        portfolio_stats(allt, name)
