"""
Challenge Donchian — rompimento 48h com assimetria positiva (prop challenge)
============================================================================
Desenhada para MAXIMIZAR P(+5%% antes de -10%%) no prop challenge, nao EV
(validacao: research/quant_lab4-6.py + quant_forward2.py, 2026-07-08):
aprovacao MC 81%% treino / 74%% ano completo / 50%% no pior regime (chop),
mediana ~12 dias, risco 0.4%%/trade.

Regras (execucao honesta, sem hipotese de fila maker):
- SINAL (candle fechado, 15m): Close rompe a maxima dos ultimos `don_n`
  candles (Donchian, excluindo o atual) na direcao da EMA de tendencia
  (Close > EMA200 para long; espelho para short).
- ENTRADA: A MERCADO na abertura do candle seguinte (taker).
- STOP: `stop_atr` x ATR14 (ewm) do candle do sinal — largo (~2-3%%),
  para o custo taker valer <0.1R. Intrabar, taker.
- ALVO: `tp_r` x risco (limite reduce-only). Fill exige ATRAVESSAR o nivel;
  no candle da entrada respeita a forma da barra (alvo so depois do fill).
- TIME-STOP: `max_bars` candles (2 semanas) no fechamento.
- Stop e alvo no MESMO candle = LOSS (conservador).
- SIZING: risco fixo por trade (default 0.4%%), alavancagem cap 5x.

O PnL emitido e BRUTO (preco), como as demais estrategias — fees e funding
ficam com o modulo de custos (entrada e stop/tempo sao TAKER; alvo maker).
"""

import sys
import os
import math
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NAME = "Challenge Donchian 48h"
DESCRIPTION = (
    "Rompimento Donchian 48h com EMA200, stop largo (8 ATR) e alvo 6R: "
    "assimetria positiva para maximizar a chance de aprovacao em prop "
    "challenge (+5/-10). Entrada a mercado, execucao conservadora."
)

CONFIG_SCHEMA = [
    {
        "title": "Sinal",
        "fields": [
            {"key": "don_n", "label": "Janela Donchian (candles)", "type": "number",
             "default": 192, "min": 24, "max": 960, "step": 24},
            {"key": "trend_len", "label": "EMA de tendência", "type": "number",
             "default": 200, "min": 50, "max": 500, "step": 10},
            {"key": "allow_shorts", "label": "Operar short", "type": "checkbox",
             "default": True},
        ],
    },
    {
        "title": "Saídas",
        "fields": [
            {"key": "stop_atr", "label": "Stop (x ATR14)", "type": "number",
             "default": 8.0, "min": 2.0, "max": 16.0, "step": 0.5},
            {"key": "tp_r", "label": "Alvo (x risco)", "type": "number",
             "default": 6.0, "min": 1.0, "max": 12.0, "step": 0.5},
            {"key": "max_bars", "label": "Time-stop (candles)", "type": "number",
             "default": 1344, "min": 96, "max": 4032, "step": 96},
        ],
    },
    {
        "title": "Sizing",
        "fields": [
            {"key": "risk_per_trade", "label": "Risco por trade (%)", "type": "number",
             "default": 0.4, "min": 0.1, "max": 5.0, "step": 0.1},
            {"key": "leverage", "label": "Alavancagem máx (x)", "type": "number",
             "default": 5.0, "min": 1.0, "max": 25.0, "step": 0.5},
        ],
    },
]

OPTIMIZER_GRIDS = {
    "rapido": {
        "don_n":    [96, 192, 384],
        "stop_atr": [5.0, 6.0, 8.0],
        "tp_r":     [4.0, 6.0, 8.0],
        "risk_per_trade": [0.4, 0.5, 0.75],
    },
}


def is_valid_config(params):
    return float(params.get("tp_r", 6.0)) >= 1.0


def prepare_optimizer_params(params):
    return params


def _safe(v):
    try:
        if v is None or math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        return v
    return float(v)


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def signal(df: pd.DataFrame, params: dict):
    """Sinal ao vivo (candles FECHADOS): ordem a MERCADO para o proximo
    candle, com politica de saida declarativa (mesma semantica do run)."""
    p = {**{"don_n": 192, "trend_len": 200, "stop_atr": 8.0, "tp_r": 6.0,
            "max_bars": 1344, "risk_per_trade": 0.4, "leverage": 5.0,
            "allow_shorts": True}, **params}
    n = int(p["don_n"])
    if len(df) < max(n, int(p["trend_len"])) + 20:
        return None
    C = df["Close"]
    H, L = df["High"], df["Low"]
    hh = H.rolling(n).max().shift(1).iloc[-1]
    ll = L.rolling(n).min().shift(1).iloc[-1]
    e = _ema(C, int(p["trend_len"])).iloc[-1]
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()],
                   axis=1).max(axis=1)
    a = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    c = C.iloc[-1]
    side = 0
    if c > hh and c > e:
        side = 1
    elif p["allow_shorts"] and c < ll and c < e:
        side = -1
    if side == 0 or not np.isfinite(a) or a <= 0:
        return None
    stop_dist_pct = float(p["stop_atr"]) * a / c * 100
    exposure = min(float(p["risk_per_trade"]) / stop_dist_pct,
                   float(p["leverage"]))
    return {
        "side": side,
        "type": "market",
        "stop_dist_pct": stop_dist_pct,
        "tp_pct": stop_dist_pct * float(p["tp_r"]),
        "sl_pct": stop_dist_pct,
        "max_bars": int(p["max_bars"]),
        "exposure": exposure,
    }


def run(df: pd.DataFrame, params: dict) -> dict:
    """Backtest honesto. df: OHLCV 15m com DatetimeIndex UTC tz-naive."""
    p = {**{"don_n": 192, "trend_len": 200, "stop_atr": 8.0, "tp_r": 6.0,
            "max_bars": 1344, "risk_per_trade": 0.4, "leverage": 5.0,
            "allow_shorts": True, "initial_capital": 1000.0}, **params}
    don_n = int(p["don_n"])
    stop_atr = float(p["stop_atr"])
    tp_r = float(p["tp_r"])
    max_bars = int(p["max_bars"])
    lev_cap = max(float(p["leverage"]), 1.0)
    risk = float(p["risk_per_trade"])
    initial_capital = float(p["initial_capital"])

    O = df["Open"].values
    H = df["High"].values
    L = df["Low"].values
    C = df["Close"].values
    N = len(df)
    TS = ((df.index - pd.Timestamp(0)).total_seconds().values * 1000).astype(np.int64)

    cs = pd.Series(C)
    hs, ls = pd.Series(H), pd.Series(L)
    hh = hs.rolling(don_n).max().shift(1).values
    ll = ls.rolling(don_n).min().shift(1).values
    e = _ema(cs, int(p["trend_len"])).values
    tr_s = pd.concat([hs - ls, (hs - cs.shift()).abs(),
                      (ls - cs.shift()).abs()], axis=1).max(axis=1)
    A = tr_s.ewm(span=14, adjust=False).mean().values

    sig = np.zeros(N)
    sig[(C > hh) & (C > e)] = 1
    if p["allow_shorts"]:
        sig[(C < ll) & (C < e)] = -1

    equity = initial_capital
    eq_curve = np.full(N, initial_capital)
    trades = []
    i = 1
    while i < N - 1:
        s = int(sig[i])
        if s == 0 or not np.isfinite(A[i]) or A[i] <= 0:
            eq_curve[i] = equity
            i += 1
            continue
        side = s
        j = i + 1
        entry = O[j]                       # mercado na abertura seguinte
        stop = entry - side * stop_atr * A[i]
        tp = entry + side * tp_r * stop_atr * A[i]
        stop_dist_pct = stop_atr * A[i] / entry * 100
        exposure = min(risk / stop_dist_pct, lev_cap) if stop_dist_pct > 0 else 0.0
        qty = exposure * equity / entry if entry > 0 else 0.0

        exit_price = None
        comment = ""
        k = j
        while k < N:
            hit_sl = (L[k] <= stop) if side == 1 else (H[k] >= stop)
            hit_tp = (H[k] > tp) if side == 1 else (L[k] < tp)
            if k == j and hit_tp:
                # forma da barra: alvo so se negociado DEPOIS da abertura
                hit_tp = (C[k] >= O[k]) if side == 1 else (C[k] < O[k])
            if hit_sl:                     # mesmo candle = LOSS
                exit_price, comment = stop, "Stop Loss"
                break
            if hit_tp:
                exit_price, comment = tp, "Alvo (6R)"
                break
            if k - j + 1 >= max_bars:
                exit_price, comment = C[k], "Time Stop"
                break
            eq_curve[k] = equity * (1 + side * (C[k] / entry - 1) * exposure)
            k += 1
        if exit_price is None:
            k = N - 1
            exit_price, comment = C[k], "Fim dos Dados"

        pnl_pct = side * (exit_price / entry - 1) * 100 * exposure
        equity *= (1 + pnl_pct / 100)
        eq_curve[k] = equity

        trades.append({
            "entry_date": str(df.index[j])[:10],
            "exit_date": str(df.index[k])[:10],
            "direction": side,
            "comment": "Breakout L" if side == 1 else "Breakout S",
            "entry_price": _safe(entry),
            "exit_price": _safe(exit_price),
            "stop_price": _safe(stop),
            "target_price": _safe(tp),
            "exit_comment": comment,
            "pnl_pct": _safe(pnl_pct),
            "partial_exit_price": None,
            "partial_exit_date": None,
            "partial_pct_closed": None,
            "qty": _safe(qty),
            "leverage": _safe(exposure),
            "notional": _safe(qty * entry),
            "entry_ts": int(TS[j]),
            "exit_ts": int(TS[k]),
        })
        i = k + 1

    # Métricas (contrato das demais estratégias)
    eq = eq_curve
    pnls = [t["pnl_pct"] for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    total_return = (equity / initial_capital - 1) * 100
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0
    pf_den = abs(sum(losses))
    profit_factor = (sum(wins) / pf_den) if pf_den > 0 else None

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak != 0, peak, 1.0) * 100
    max_dd = float(dd.min()) if len(dd) else 0.0
    total_days = max((df.index[-1] - df.index[0]).days, 1) if N > 1 else 1
    cagr = ((equity / initial_capital) ** (365.25 / total_days) - 1) * 100
    dates = [str(idx)[:10] for idx in df.index]

    return {
        "chart": None,
        "metrics": {
            "final_equity": _safe(equity),
            "total_return": _safe(total_return),
            "max_dd": _safe(max_dd),
            "total_trades": len(trades),
            "win_rate": _safe(win_rate),
            "profit_factor": _safe(profit_factor),
            "avg_win": _safe(float(np.mean(wins))) if wins else 0.0,
            "avg_loss": _safe(float(np.mean(losses))) if losses else 0.0,
            "sharpe": None,
            "sortino": None,
            "calmar": _safe(cagr / abs(max_dd)) if max_dd < 0 else None,
            "omega": None,
            "sterling": None,
            "burke": None,
            "recovery_factor": None,
            "ulcer_index": None,
            "avg_dd": None,
            "avg_dd_length": None,
            "n_dd_episodes": 0,
            "initial_capital": initial_capital,
        },
        "drawdown": {"dates": dates,
                     "values": [_safe(v) for v in dd.tolist()],
                     "episodes": []},
        "equity_curve": {"dates": dates,
                         "values": [_safe(v) for v in eq.tolist()]},
        "trades": trades,
    }
