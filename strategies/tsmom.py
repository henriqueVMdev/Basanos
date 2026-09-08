"""
TSMOM Ensemble — momentum diário com regime BTC e vol targeting
================================================================
Estratégia DIÁRIA validada em research/daily_opt.py (2026-07-09) com
protocolo anti-overfit (treino <2024 / validação 24-25 uma passada /
holdout 2026):

  treino <2024:  PF 2.75  sharpe ~2.2  (positivo em TODOS os anos, incl. 2022)
  validação 24-25 (1 passada):  PF 1.64  sharpe ~1.27  (+1.0%/trade)

TSMOM puro (um único lookback) NÃO sobrevive aos custos em 2025-26 — o
edge só aparece com as três peças estruturais juntas, nenhuma tunada por
símbolo:

 1) ENSEMBLE de lookbacks: voto de sinal do retorno de 7/14/21/28 dias;
    média dos votos >= +0.5 -> long; <= -0.5 -> short (remove a escolha
    arbitrária de um único n).
 2) REGIME de mercado: BTC > SMA200 diária -> só longs; BTC < SMA200 ->
    só shorts (conserta os anos de bear, ex.: 2022/2025).
 3) VOL TARGETING: exposição = min(3.0 / ATR%%, 1.5) — arrisca menos
    quando a vol está alta (regra clássica, não tunada).

Execução (mesma honestidade do research):
- Sinal no FECHAMENTO diário; entrada a MERCADO na abertura do dia
  seguinte (taker). Posicional: mantém a posição enquanto o lado
  desejado não muda; inverte/sai na abertura seguinte à virada.
- Sem TP/SL/time-stop (saída só por virada de sinal).

O PnL emitido por run() é BRUTO (preço x exposição), como as demais
estratégias — fees (taker nas duas pontas) e funding (drag ~0.03%%/dia no
long) ficam com o módulo de custos. Os números NET do research já
descontam isso.

Gêmea de automação: strategies/daily_ensemble.py expõe o mesmo signal()
e está em paper (13 símbolos). Espelho TradingView: strategies/tsmom.pine.
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NAME = "TSMOM Ensemble (diário)"
DESCRIPTION = (
    "Momentum diário: ensemble de lookbacks (7/14/21/28) + regime BTC/SMA200 "
    "(bull=long, bear=short) + vol targeting. Validado com holdout; posicional, "
    "entrada a mercado."
)

_DEFAULTS = {
    "ns": (7, 14, 21, 28),
    "thr": 0.5,
    "regime_ma": 200,
    "target_atrpct": 3.0,
    "w_cap": 1.5,
    "allow_shorts": True,
    "initial_capital": 1000.0,
}

CONFIG_SCHEMA = [
    {
        "title": "Sinal",
        "fields": [
            {"key": "thr", "label": "Limiar do ensemble", "type": "number",
             "default": 0.5, "min": 0.25, "max": 1.0, "step": 0.25},
            {"key": "regime_ma", "label": "SMA de regime (BTC)", "type": "number",
             "default": 200, "min": 100, "max": 300, "step": 10},
            {"key": "allow_shorts", "label": "Operar short (bear)", "type": "checkbox",
             "default": True},
        ],
    },
    {
        "title": "Sizing",
        "fields": [
            {"key": "target_atrpct", "label": "ATR%% alvo (vol target)", "type": "number",
             "default": 3.0, "min": 1.0, "max": 6.0, "step": 0.5},
            {"key": "w_cap", "label": "Exposição máx (x)", "type": "number",
             "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.25},
        ],
    },
]

OPTIMIZER_GRIDS = {
    "rapido": {
        "thr": [0.25, 0.5],
        "regime_ma": [150, 200, 250],
        "target_atrpct": [2.0, 3.0, 4.0],
    },
}


def is_valid_config(params):
    return float(params.get("thr", 0.5)) >= 0.25


def prepare_optimizer_params(params):
    return params


def _safe(v):
    try:
        if v is None or not np.isfinite(v):
            return None
    except (TypeError, ValueError):
        return v
    return float(v)


# indicadores
def _atr(H, L, C, n=14):
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _ensemble_side(C: pd.Series, ns, thr) -> np.ndarray:
    """Voto médio dos lookbacks -> {-1,0,1} bruto (antes do regime).
    Fiel ao research (sig_tsmom): voto = +1 se retorno de n dias > 0 senão
    -1; região sem histórico do lookback vota 0."""
    votes = np.zeros(len(C))
    for n in ns:
        r = C.pct_change(n).values
        v = np.where(r > 0, 1.0, -1.0)
        v[np.isnan(r)] = 0.0
        votes += v
    m = votes / len(ns)
    return np.where(m >= thr, 1.0, np.where(m <= -thr, -1.0, 0.0))


def _btc_bull(index: pd.DatetimeIndex, regime_ma: int) -> np.ndarray:
    """Série booleana BTC>SMA200 alinhada ao índice do símbolo (ffill).
    Busca BTC 1d via market_data; sem rede/dado, cai para NaN (sinal 0)."""
    try:
        from providers import market_data
        btc = market_data.fetch_ohlcv("BTC", "1d", exchange="binance",
                                      total=len(index) + regime_ma + 50)
        c = btc["Close"]
        bull = (c > c.rolling(regime_ma).mean())
        return bull.reindex(index, method="ffill").values
    except Exception:
        return np.full(len(index), np.nan)


def _gated_signal(df: pd.DataFrame, p: dict) -> np.ndarray:
    """Sinal posicional final: ensemble filtrado pelo regime BTC."""
    raw = _ensemble_side(df["Close"], tuple(p["ns"]), float(p["thr"]))
    bull = _btc_bull(df.index, int(p["regime_ma"]))
    out = np.zeros(len(df))
    long_ok = (bull == 1) & (raw > 0)
    short_ok = (bull == 0) & (raw < 0) & bool(p["allow_shorts"])
    out[long_ok] = 1.0
    out[short_ok] = -1.0
    out[np.isnan(bull)] = 0.0
    return out


# sinal ao vivo (contrato da automação)
def signal(df: pd.DataFrame, params: dict):
    """Candles DIÁRIOS fechados -> posição desejada para a próxima abertura
    (market/open) ou None. Posicional: exit_on_flip."""
    p = {**_DEFAULTS, **(params or {})}
    if len(df) < 400:
        return None
    s = _gated_signal(df, p)
    side = int(s[-1])
    if side == 0:
        return None
    atr_pct = float(_atr(df["High"], df["Low"], df["Close"]).iloc[-1]) \
        / float(df["Close"].iloc[-1]) * 100
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
        "exposure": min(float(p["target_atrpct"]) / atr_pct, float(p["w_cap"])),
        "exit_on_flip": True,
    }


# backtest
def run(df: pd.DataFrame, params: dict) -> dict:
    """Backtest honesto posicional. df: OHLCV DIÁRIO, DatetimeIndex UTC.
    PnL BRUTO (preço x exposição); fees/funding ficam com o módulo de custos."""
    p = {**_DEFAULTS, **(params or {})}
    initial_capital = float(p["initial_capital"])
    target, w_cap = float(p["target_atrpct"]), float(p["w_cap"])

    O = df["Open"].values
    C = df["Close"].values
    N = len(df)
    TS = ((df.index - pd.Timestamp(0)).total_seconds().values * 1000).astype(np.int64)
    A = _atr(df["High"], df["Low"], df["Close"]).values
    sig = _gated_signal(df, p)

    equity = initial_capital
    eq_curve = np.full(N, initial_capital, dtype=float)
    trades = []

    pos = 0
    entry = 0.0
    exposure = 0.0
    j0 = -1
    for i in range(1, N - 1):
        s = int(sig[i])
        if s != pos:
            if pos != 0:                       # fecha na abertura de i+1
                exit_p = O[i + 1]
                pnl_pct = pos * (exit_p / entry - 1) * 100 * exposure
                equity *= (1 + pnl_pct / 100)
                eq_curve[i + 1] = equity
                trades.append(_mk_trade(df, TS, pos, entry, exit_p, exposure,
                                        pnl_pct, j0, i + 1, "Sinal contrário"))
                pos = 0
            if s != 0 and np.isfinite(A[i]) and A[i] > 0:
                entry = O[i + 1]
                atr_pct = A[i] / C[i] * 100
                exposure = min(target / atr_pct, w_cap) if atr_pct > 0 else 0.0
                pos = s
                j0 = i + 1
        if pos != 0:
            eq_curve[i] = equity * (1 + pos * (C[i] / entry - 1) * exposure)
        else:
            eq_curve[i] = equity
    if pos != 0:                               # fecha aberto no fim
        exit_p = C[N - 1]
        pnl_pct = pos * (exit_p / entry - 1) * 100 * exposure
        equity *= (1 + pnl_pct / 100)
        eq_curve[N - 1] = equity
        trades.append(_mk_trade(df, TS, pos, entry, exit_p, exposure,
                                pnl_pct, j0, N - 1, "Fim dos Dados"))

    return _report(df, trades, eq_curve, equity, initial_capital)


def _mk_trade(df, TS, side, entry, exit_p, exposure, pnl_pct, j0, j1, comment):
    return {
        "entry_date": str(df.index[j0])[:10],
        "exit_date": str(df.index[j1])[:10],
        "direction": side,
        "comment": "TSMOM L" if side == 1 else "TSMOM S",
        "entry_price": _safe(entry),
        "exit_price": _safe(exit_p),
        "stop_price": None,
        "target_price": None,
        "exit_comment": comment,
        "pnl_pct": _safe(pnl_pct),
        "partial_exit_price": None,
        "partial_exit_date": None,
        "partial_pct_closed": None,
        "qty": _safe(exposure * entry and (exposure / entry)),
        "leverage": _safe(exposure),
        "notional": _safe(exposure),
        "entry_ts": int(TS[j0]),
        "exit_ts": int(TS[j1]),
    }


def _report(df, trades, eq_curve, equity, initial_capital):
    N = len(df)
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
            "sharpe": None, "sortino": None,
            "calmar": _safe(cagr / abs(max_dd)) if max_dd < 0 else None,
            "omega": None, "sterling": None, "burke": None,
            "recovery_factor": None, "ulcer_index": None,
            "avg_dd": None, "avg_dd_length": None, "n_dd_episodes": 0,
            "initial_capital": initial_capital,
        },
        "drawdown": {"dates": dates,
                     "values": [_safe(v) for v in dd.tolist()], "episodes": []},
        "equity_curve": {"dates": dates,
                         "values": [_safe(v) for v in eq.tolist()]},
        "trades": trades,
    }
