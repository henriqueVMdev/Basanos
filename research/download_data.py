# -*- coding: utf-8 -*-
"""Baixa os dados usados pela pesquisa: 1 ano de candles 15m + funding
de 12 perps USDT da Bybit, para research/data/. Rode a partir da raiz do
projeto (precisa de market_data.py e costs/ no PYTHONPATH)."""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers.market_data import fetch_ohlcv          # noqa: E402
from costs.funding import get_funding_events  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

SYMS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK",
        "AVAX", "SUI", "LTC", "NEAR"]

for sym in SYMS:
    df = fetch_ohlcv(sym, "15m", exchange="bybit", total=35000)
    df.to_csv(os.path.join(DATA, f"{sym.lower()}_15m_bybit.csv"))
    since = int((df.index[0] - pd.Timestamp(0)).total_seconds() * 1000)
    until = int((df.index[-1] - pd.Timestamp(0)).total_seconds() * 1000)
    ev = get_funding_events("bybit", f"{sym}/USDT:USDT", since, until)
    pd.DataFrame({"timestamp": [e.timestamp for e in ev],
                  "rate": [float(e.rate) for e in ev]}
                 ).to_csv(os.path.join(DATA, f"funding_{sym.lower()}.csv"), index=False)
    print(sym, df.shape[0], "candles | funding:", len(ev), flush=True)
    time.sleep(5)  # cortesia com o rate limit da Bybit
print("FIM")
