# -*- coding: utf-8 -*-
"""Baixa candles DIARIOS da Binance spot (historico desde 2017 para os
majors) para a pesquisa de estrategia no diario. Salva {sym}_1d_binance.csv.

Uso: py research/download_daily.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers.market_data import fetch_ohlcv, get_exchange  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
get_exchange("binance").enableRateLimit = True

SYMS = ["BTC", "ETH", "BNB", "LTC", "XRP", "ADA", "LINK", "DOGE",
        "SOL", "AVAX", "DOT", "NEAR", "SUI", "MATIC"]

for sym in SYMS:
    out = os.path.join(DATA, f"{sym.lower()}_1d_binance.csv")
    if os.path.exists(out):
        print(sym, "ja existe — pulado", flush=True)
        continue
    try:
        df = fetch_ohlcv(sym, "1d", exchange="binance", total=3500)
    except Exception as e:
        print(sym, "ERRO:", e, flush=True)
        continue
    df.to_csv(out)
    print(sym, df.shape[0], "dias |", str(df.index[0])[:10], "->",
          str(df.index[-1])[:10], flush=True)
    time.sleep(2)
print("FIM")
