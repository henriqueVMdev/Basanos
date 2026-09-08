# -*- coding: utf-8 -*-
"""Compara TV x local na MESMA janela do Strategy Tester (Feb 28 2026 ->
Jul 8 2026, BYBIT BTCUSDT.P 15m), em tres versoes:

  1. APP local (semantica do repo, PnL bruto)
  2. APP local com comissao 0.025%/ponta
  3. TV EMULADO (fill no toque, gap na abertura, TP+SL mesmo candle pela
     forma da barra, sessao no candle do setup, percentrank do Pine,
     comissao embutida) -- se este bater com o tester, a divergencia esta
     100% explicada pela semantica.

Referencia do tester TV nesta janela (instancia nova, defaults):
  trades 83 | WR 74.7% | PF 0.988 | retorno -0.24% | DD 6.30%

Uso: py research/parity_tv_window.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from providers.market_data import fetch_ohlcv
import strategies.mm9_pullback as mm9

START, END = "2026-02-28", "2026-07-08 23:59"
TP, SL, MAX_BARS = 0.5, 1.5, 48
COMMISSION_SIDE = 0.025 / 100
VOL_RANK_MIN = 0.5
EXPOSURE = min(1.0 / SL, 2.0)

print("Baixando BTC 15m (Bybit)...")
df = fetch_ohlcv("BTC", "15m", "bybit", limit=1000, total=15000)
df = df[(df.index >= START) & (df.index <= END)]
df = df.iloc[:-1]  # descarta a vela possivelmente em formacao
print(f"janela: {df.index[0]} -> {df.index[-1]} | {len(df)} barras")

O = df["Open"].values
H = df["High"].values
L = df["Low"].values
C = df["Close"].values
N = len(df)

feat = mm9._compute_features(df, {})
e9 = feat["e9"]
long_ok_raw, short_ok_raw = feat["long_ok"], feat["short_ok"]

# percentrank estilo Pine: % das 384 barras ANTERIORES com valor <= atual
close_s = pd.Series(C, index=df.index)
h_s = pd.Series(H, index=df.index)
l_s = pd.Series(L, index=df.index)
tr = pd.concat([h_s - l_s, (h_s - close_s.shift()).abs(),
                (l_s - close_s.shift()).abs()], axis=1).max(axis=1)
atr_pct = (tr.ewm(span=14, adjust=False).mean() / close_s * 100).values

pine_rank = np.full(N, np.nan)
for i in range(384, N):
    w = atr_pct[i - 384:i]
    pine_rank[i] = (w <= atr_pct[i]).mean()
vol_ok_pine = pine_rank >= VOL_RANK_MIN

# Sessao 19h-00h Sao Paulo (UTC-3) avaliada no candle do SETUP (pine)
local_hour = (df.index.hour.values - 3) % 24
in_session_setup = local_hour >= 19


def run_tv_emulation():
    equity = 50_000.0
    trades = []
    i = 1
    while i < N:
        s = i - 1
        setup_long = long_ok_raw[s] and vol_ok_pine[s] and in_session_setup[s]
        setup_short = short_ok_raw[s] and vol_ok_pine[s] and in_session_setup[s]
        if not (setup_long or setup_short):
            i += 1
            continue
        side = 1 if setup_long else -1
        limit = e9[s]

        # fill TV: toque; gap preenche na abertura (preco melhor)
        if side == 1:
            if O[i] <= limit:
                entry = O[i]
            elif L[i] <= limit:
                entry = limit
            else:
                i += 1
                continue
        else:
            if O[i] >= limit:
                entry = O[i]
            elif H[i] >= limit:
                entry = limit
            else:
                i += 1
                continue

        exit_price = None
        comment = ""
        j = i
        while j < N:
            anchor = limit if j == i else entry
            tp = anchor * (1 + TP / 100 * side)
            sl = anchor * (1 - SL / 100 * side)
            hit_tp = (H[j] >= tp) if side == 1 else (L[j] <= tp)
            hit_sl = (L[j] <= sl) if side == 1 else (H[j] >= sl)
            if hit_tp and hit_sl:
                green = C[j] >= O[j]
                first_tp = (not green) if side == 1 else green
                exit_price = tp if first_tp else sl
                comment = "TP(shape)" if first_tp else "SL(shape)"
                break
            if hit_sl:
                exit_price, comment = sl, "SL"
                break
            if hit_tp:
                exit_price, comment = tp, "TP"
                break
            if j - i >= MAX_BARS - 1:
                exit_price, comment = C[j], "TIME"
                break
            j += 1
        if exit_price is None:
            j = N - 1
            exit_price, comment = C[j], "EOD"

        move = side * (exit_price / entry - 1) * 100
        pnl = EXPOSURE * move - 2 * COMMISSION_SIDE * 100 * EXPOSURE
        equity *= (1 + pnl / 100)
        trades.append({"pnl": pnl, "comment": comment,
                       "entry_ts": str(df.index[i])})
        i = j + 1
    return equity, trades


def stats(pnls, label, eq0=50_000.0, eq1=None):
    wins = [p for p in pnls if p > 0]
    pf_den = abs(sum(p for p in pnls if p <= 0))
    pf = sum(wins) / pf_den if pf_den else float("nan")
    ret = f" | retorno {(eq1 / eq0 - 1) * 100:+.2f}%" if eq1 else ""
    print(f"{label:36s} | trades {len(pnls):3d} | WR {len(wins)/len(pnls)*100:5.1f}% "
          f"| PF {pf:5.2f}{ret}")


res = mm9.run(df, {"initial_capital": 50_000.0})
app_pnls = [t["pnl_pct"] for t in res["trades"]]
stats(app_pnls, "APP local (bruto)", eq1=res["metrics"]["final_equity"])

cost = 2 * COMMISSION_SIDE * 100 * EXPOSURE
app_net = [p - cost for p in app_pnls]
eq = 50_000.0
for p in app_net:
    eq *= 1 + p / 100
stats(app_net, "APP local c/ comissao", eq1=eq)

eq_tv, tv_trades = run_tv_emulation()
stats([t["pnl"] for t in tv_trades], "TV EMULADO (toque+shape+comissao)", eq1=eq_tv)
print("saidas TV emulado:", dict(Counter(t["comment"] for t in tv_trades)))
print()
print("TV TESTER (referencia)                | trades  83 | WR  74.7% | PF  0.99 | retorno -0.24%")
