# -*- coding: utf-8 -*-
"""
Daily Ensemble — TSMOM ensemble + regime BTC + vol targeting (POSICIONAL)
=========================================================================
Pacote validado em research/daily_opt.py (2026-07-08):
- ensemble TSMOM: média dos votos de retorno de 7/14/21/28 dias
  (média >= +0.5 -> long; <= -0.5 -> short);
- filtro de regime: BTC > SMA200 diária -> só longs; abaixo -> só shorts;
- vol targeting: exposição = min(3.0 / ATR%, 1.5).

Posicional: entra a MERCADO na abertura do dia seguinte ao sinal e sai
quando o lado desejado muda (exit_on_flip) — sem TP/SL/time-stop.

Premissas do paper vs backtest do research: o paper não simula slippage
nem funding (o backtest usou 0.05% de slip por perna + drag de funding),
logo o paper é ~0.1%/rodada + funding OTIMISTA. Números de referência:
treino <2024 PF 2.75 | validação 24-25 PF 1.64 | forward pendente.

Módulo só de automação (signal()); não expõe run() para o frontend.
"""

import numpy as np
import pandas as pd

NAME = "Daily Ensemble TSMOM"
DESCRIPTION = ("Ensemble TSMOM diário + regime BTC/SMA200 + vol targeting "
               "(posicional, paper)")

_DEFAULTS = {
    "ns": (7, 14, 21, 28),
    "thr": 0.5,
    "regime_ma": 200,
    "target_atrpct": 3.0,
    "w_cap": 1.5,
}

_regime_cache = None


def _fetch_regime(regime_ma: int) -> pd.DataFrame:
    from providers import market_data
    return market_data.fetch_ohlcv("BTC", "1d", exchange="binance",
                                   total=regime_ma + 250)


def _regime_closes(last_ts, regime_ma: int) -> pd.Series:
    """Fechamentos diários de BTC ATÉ last_ts (corta lookahead no replay).
    ponytail: o fetch cobre ~250 dias de gap de replay; gap maior deixa o
    regime insuficiente e o sinal fica None até o histórico normalizar."""
    global _regime_cache
    if _regime_cache is None or _regime_cache.index[-1] < last_ts:
        _regime_cache = _fetch_regime(regime_ma)
    return _regime_cache["Close"][_regime_cache.index <= last_ts]


def signal(df: pd.DataFrame, params: dict):
    """df: candles DIÁRIOS fechados. Retorna a posição desejada para a
    próxima abertura (market/open) ou None (flat)."""
    p = {**_DEFAULTS, **(params or {})}
    if len(df) < 400:
        return None

    C = df["Close"]
    votes = [1.0 if float(C.iloc[-1]) > float(C.iloc[-1 - int(n)]) else -1.0
             for n in p["ns"]]
    m = float(np.mean(votes))
    side = 1 if m >= p["thr"] else (-1 if m <= -p["thr"] else 0)
    if side == 0:
        return None

    regime_ma = int(p["regime_ma"])
    btc = _regime_closes(df.index[-1], regime_ma)
    if len(btc) < regime_ma:
        return None
    bull = float(btc.iloc[-1]) > float(btc.rolling(regime_ma).mean().iloc[-1])
    if (side == 1) != bull:                 # long só em bull, short só em bear
        return None

    H, L = df["High"], df["Low"]
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()],
                   axis=1).max(axis=1)
    a = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
    atr_pct = a / float(C.iloc[-1]) * 100
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return None

    return {
        "side": side,
        "type": "market",
        "price": None,
        "valid_bars": 1,
        "tp_pct": None,
        "sl_pct": None,
        "max_bars": None,
        "fill_rule": "open",
        "exposure": min(p["target_atrpct"] / atr_pct, p["w_cap"]),
        "exit_on_flip": True,
    }
