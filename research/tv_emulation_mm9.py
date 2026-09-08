# -*- coding: utf-8 -*-
"""
Emula o backtest do TRADINGVIEW para a MM9 Pullback Maker sobre os MESMOS
dados (Bybit 15m via ccxt) e compara com o backtest do app, isolando cada
diferença de semântica:

  TV (emulado)                          | App (Python)
  --------------------------------------+---------------------------------
  fill de limite no TOQUE (low <= lim)  | exige ATRAVESSAR (low < lim)
  gap: fill na ABERTURA (preço melhor)  | fill sempre no preço da limite
  TP no toque (high >= tp)              | exige atravessar (high > tp)
  TP+SL no mesmo candle: forma da barra | sempre LOSS (conservador)
    (barra verde O→L→H→C; vermelha O→H→L→C)
  janela de horário no candle do SETUP  | no candle do FILL
  percentrank exclui o candle atual     | rank pandas inclui o atual
  comissão 0.025%/ponta embutida        | bruto (custos à parte)

Uso:  py research/tv_emulation_mm9.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from providers.market_data import fetch_ohlcv
import strategies.mm9_pullback as mm9

# ── parâmetros (defaults da estratégia / do .pine) ───────────────────────
TP, SL, MAX_BARS = 0.5, 1.5, 48
RISK, LEV_CAP = 1.0, 2.0
COMMISSION_SIDE = 0.025 / 100          # 0.025% por ponta (pine atual)
VOL_RANK_MIN = 0.5
EXPOSURE = min(RISK / SL, LEV_CAP)     # 0.667x

print("Baixando BTC 15m (1 ano, Bybit)...")
df = fetch_ohlcv("BTC", "15m", "bybit", limit=1000, total=35040)
print(f"janela: {df.index[0]} -> {df.index[-1]} | {len(df)} barras")

O = df["Open"].values
H = df["High"].values
L = df["Low"].values
C = df["Close"].values
N = len(df)

# Filtros idênticos aos do app (EMAs, slope 1h sem lookahead, direções)
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

# Sessão: hora LOCAL (UTC-3) do candle do SETUP em {19..23} (pine)
HOUR_UTC = df.index.hour.values
local_hour = (HOUR_UTC - 3) % 24
in_session_setup = (local_hour >= 19)  # 19..23 local


def run_tv_emulation():
    equity = 50_000.0
    trades = []
    i = 1
    while i < N:
        s = i - 1                                   # candle do setup
        setup_long = long_ok_raw[s] and vol_ok_pine[s] and in_session_setup[s]
        setup_short = short_ok_raw[s] and vol_ok_pine[s] and in_session_setup[s]
        if not (setup_long or setup_short):
            i += 1
            continue
        side = 1 if setup_long else -1
        limit = e9[s]

        # fill TV: toque; gap preenche na abertura (preço melhor)
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

        # bracket: no candle do fill ancorado na LIMITE (pine); depois no
        # preço médio real
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
                # ambidestro: emulador do TV decide pela forma da barra
                green = C[j] >= O[j]
                if side == 1:
                    first_tp = not green      # vermelha: O→H→L→C (topo antes)
                else:
                    first_tp = green          # verde: O→L→H→C (fundo antes)
                exit_price = tp if first_tp else sl
                comment = "TP(shape)" if first_tp else "SL(shape)"
                break
            if hit_sl:
                exit_price, comment = sl, "SL"
                break
            if hit_tp:
                exit_price, comment = tp, "TP"
                break
            if j - i >= MAX_BARS - 1:          # time-stop no close (pine novo)
                exit_price, comment = C[j], "TIME"
                break
            j += 1
        if exit_price is None:
            j = N - 1
            exit_price, comment = C[j], "EOD"

        move = side * (exit_price / entry - 1) * 100
        pnl = EXPOSURE * move - 2 * COMMISSION_SIDE * 100 * EXPOSURE
        equity *= (1 + pnl / 100)
        trades.append({"pnl": pnl, "comment": comment})
        i = j + 1
    return equity, trades


def stats(pnls, label, eq0=50_000.0, eq1=None):
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
    ret = (eq1 / eq0 - 1) * 100 if eq1 else None
    print(f"{label:34s} | trades {len(pnls):4d} | WR {len(wins)/len(pnls)*100:5.1f}% "
          f"| PF {pf:5.2f} | média {np.mean(pnls):+.4f}%/trade"
          + (f" | retorno {ret:+.1f}%" if ret is not None else ""))


# ── 1) App (bruto e com comissão do TV p/ isolar custo) ─────────────────
res = mm9.run(df, {"initial_capital": 50_000.0})
app_pnls = [t["pnl_pct"] for t in res["trades"]]
app_eq = res["metrics"]["final_equity"]
stats(app_pnls, "APP (bruto, como na tela)", eq1=app_eq)

cost = 2 * COMMISSION_SIDE * 100 * EXPOSURE
app_net = [p - cost for p in app_pnls]
eq = 50_000.0
for p in app_net:
    eq *= 1 + p / 100
stats(app_net, "APP c/ comissao 0.025%/ponta", eq1=eq)

# ── 2) TV emulado ────────────────────────────────────────────────────────
eq_tv, tv_trades = run_tv_emulation()
tv_pnls = [t["pnl"] for t in tv_trades]
stats(tv_pnls, "TV EMULADO (toque+shape+comissao)", eq1=eq_tv)

from collections import Counter
print("saidas TV emulado:", dict(Counter(t["comment"] for t in tv_trades)))
