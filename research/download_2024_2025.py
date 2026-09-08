# -*- coding: utf-8 -*-
"""Baixa 2024-2025 (15m + funding) dos 12 perps Bybit para a validacao
multi-regime do Challenge Donchian. Inclui warm-up desde 2023-12-15.
Salva como {sym}_15m_bybit_2425.csv / funding_{sym}_2425.csv.

Uso: py research/download_2024_2025.py"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers.market_data import fetch_ohlcv, get_exchange  # noqa: E402
from costs.funding import get_funding_events        # noqa: E402
from ccxt.base.errors import RateLimitExceeded      # noqa: E402

# espaca as requests automaticamente (a paginacao dispara ~72 seguidas)
get_exchange("bybit").enableRateLimit = True

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SYMS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK",
        "AVAX", "SUI", "LTC", "NEAR"]
START, END = "2023-12-15", "2026-01-01"

print("cooldown de 90s (rate limit Bybit)...", flush=True)
time.sleep(90)

for sym in SYMS:
    out = os.path.join(DATA, f"{sym.lower()}_15m_bybit_2425.csv")
    if os.path.exists(out):
        print(sym, "ja existe — pulado", flush=True)
        continue
    # total suficiente para alcancar 2023-12-15 a partir de hoje (2026-07);
    # retry com backoff no rate limit
    for attempt in range(5):
        try:
            df = fetch_ohlcv(sym, "15m", exchange="bybit", total=92000)
            break
        except RateLimitExceeded:
            wait = 120 * (attempt + 1)
            print(f"{sym}: rate limit — aguardando {wait}s", flush=True)
            time.sleep(wait)
    else:
        print(f"{sym}: desistindo apos 5 tentativas", flush=True)
        continue
    df = df[(df.index >= START) & (df.index < END)]
    df.to_csv(out)
    since = int((df.index[0] - pd.Timestamp(0)).total_seconds() * 1000)
    until = int((df.index[-1] - pd.Timestamp(0)).total_seconds() * 1000)
    ev = get_funding_events("bybit", f"{sym}/USDT:USDT", since, until)
    pd.DataFrame({"timestamp": [e.timestamp for e in ev],
                  "rate": [float(e.rate) for e in ev]}
                 ).to_csv(os.path.join(DATA, f"funding_{sym.lower()}_2425.csv"),
                          index=False)
    print(sym, df.shape[0], "candles | funding:", len(ev), flush=True)
    time.sleep(30)
print("FIM")
