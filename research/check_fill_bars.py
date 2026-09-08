# -*- coding: utf-8 -*-
"""Confere a forma do candle do fill nos 3 trades que divergem TV x local."""
import sys
sys.path.insert(0, '.')
from providers.market_data import fetch_ohlcv

df = fetch_ohlcv('BTC', '15m', 'bybit', limit=1000, total=15000)
cases = [('2026-05-11 00:00', 1, 81895.1),
         ('2026-06-04 02:15', -1, 63014.9),
         ('2026-06-29 02:15', -1, 59344.0)]
for ts, side, entry in cases:
    bar = df.loc[ts]
    tp = entry * (1 + 0.005 * side)
    hit_tp = bar['High'] >= tp if side == 1 else bar['Low'] <= tp
    green = bar['Close'] >= bar['Open']
    s = 'L' if side == 1 else 'S'
    print(f"{ts} {s}: O={bar['Open']:.0f} H={bar['High']:.0f} "
          f"L={bar['Low']:.0f} C={bar['Close']:.0f} | limite={entry:.0f} "
          f"tp={tp:.0f} | TP tocado no candle do fill: {hit_tp} | "
          f"candle {'verde' if green else 'vermelho'}")
