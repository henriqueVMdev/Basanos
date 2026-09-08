"""
Engine de paper trading dirigido por CANDLE FECHADO — nunca por relógio.

Semântica idêntica ao backtest (strategies/mm9_pullback.run):
- Ordem limite vale por 1 candle; fill só se o preço ATRAVESSAR
  (Low < limite p/ long, High > limite p/ short — estrito).
- No candle da posição: SL primeiro (Low<=sl / High>=sl), depois TP
  (High>tp / Low<tp — estrito, limite maker), depois time-stop no Close.
  TP e SL no mesmo candle = LOSS (SL testado antes). TP no candle do
  fill só vale se a forma da barra permite (long+verde/short+vermelho).
- pnl_pct escalado pela exposição (mesma fórmula do backtest).

Fees (opcional, default ligado): maker na entrada e no TP, taker no SL e
no time-stop — tabela da Bybit em costs/config.py. Funding NÃO é simulado
no paper (registrado como premissa; o modo demo tem funding real).

Como o processamento é por candle fechado em ordem, desligar o PC e
religar produz replay determinístico: mesmo resultado de ter ficado ligado.
"""

from __future__ import annotations

from decimal import Decimal

try:
    from costs.config import DEFAULT_FEES
    _F = DEFAULT_FEES["bybit"]
    MAKER, TAKER = float(_F.maker), float(_F.taker)
except Exception:                      # fallback: tabela 2026 da Bybit
    MAKER, TAKER = 0.0002, 0.00055


def check_entry_fill(order: dict, candle: dict) -> bool:
    """Limite 'cross': o candle atravessou o preço? Market: sempre preenche
    no candle em que vale (na abertura)."""
    if candle["ts"] != order["valid_candle_ts"]:
        return False
    if order["type"] == "market":
        return True
    if order["side"] == 1:
        return candle["low"] < order["price"]
    return candle["high"] > order["price"]


def open_position_from_order(order: dict, equity: float, candle_ts: int,
                             fill_price: float | None = None) -> dict:
    """Monta a posição a partir da ordem preenchida (sizing do backtest).
    fill_price: preço de fill p/ ordens market (abertura do candle);
    limit usa o preço da ordem. tp/sl/max_bars None = sem essa saída."""
    entry = float(fill_price if fill_price is not None else order["price"])
    side = int(order["side"])
    exposure = float(order["exposure"])
    qty = exposure * equity / entry if entry > 0 else 0.0
    return {
        "side": side,
        "qty": qty,
        "exposure": exposure,
        "entry_price": entry,
        "entry_candle_ts": candle_ts,
        "tp_price": entry * (1 + float(order["tp_pct"]) / 100 * side)
                    if order["tp_pct"] is not None else None,
        "sl_price": entry * (1 - float(order["sl_pct"]) / 100 * side)
                    if order["sl_pct"] is not None else None,
        "max_bars": int(order["max_bars"]) if order["max_bars"] is not None else None,
        "bars_held": 0,
        "entry_fee_rate": TAKER if order["type"] == "market" else MAKER,
        "exit_on_flip": 1 if (order.get("raw") or {}).get("exit_on_flip") else 0,
    }


def check_exit(pos: dict, candle: dict) -> dict | None:
    """Avalia saída da posição neste candle (ordem: SL, TP, time-stop).

    bars_held do candle de entrada = 1 (mesmo candle pode estopar, igual
    ao backtest onde j começa em i). Retorna None ou
    {exit_price, reason, exit_fee_rate}.
    """
    side = pos["side"]
    sl, tp = pos["sl_price"], pos["tp_price"]
    hit_sl = sl is not None and (
        (candle["low"] <= sl) if side == 1 else (candle["high"] >= sl))
    hit_tp = tp is not None and (
        (candle["high"] > tp) if side == 1 else (candle["low"] < tp))
    if hit_tp and candle["ts"] == pos["entry_candle_ts"]:
        # TP no candle do fill segue a forma da barra: o alvo precisa ser
        # negociado DEPOIS do fill na trajetória intrabar (long exige candle
        # verde, short vermelho) — mesma regra do backtest (mm9_pullback.run)
        hit_tp = (candle["close"] >= candle["open"]) if side == 1 \
            else (candle["close"] < candle["open"])
    if hit_sl:
        return {"exit_price": sl, "reason": "Stop Loss", "exit_fee_rate": TAKER}
    if hit_tp:
        return {"exit_price": tp, "reason": "Alvo (maker)", "exit_fee_rate": MAKER}
    if pos["max_bars"] is not None and pos["bars_held"] + 1 >= pos["max_bars"]:
        return {"exit_price": candle["close"], "reason": "Time Stop",
                "exit_fee_rate": TAKER}
    return None


def close_pnl(pos: dict, exit_price: float, exit_fee_rate: float,
              equity: float, with_fees: bool = True) -> dict:
    """PnL do fechamento, na mesma base do backtest (% do equity de entrada).

    Fees sobre o notional (entrada maker + saída conforme o gatilho),
    convertidas para % do equity via exposição.
    """
    side, exposure = pos["side"], pos["exposure"]
    gross_pct = side * (exit_price / pos["entry_price"] - 1) * 100 * exposure
    entry_fee = pos.get("entry_fee_rate") or MAKER
    fees_pct = ((entry_fee + exit_fee_rate) * 100 * exposure) if with_fees else 0.0
    pnl_pct = gross_pct - fees_pct
    pnl_quote = equity * pnl_pct / 100
    fees_quote = equity * fees_pct / 100
    return {"pnl_pct": pnl_pct, "gross_pct": gross_pct,
            "pnl_quote": pnl_quote, "fees_quote": fees_quote,
            "new_equity": equity * (1 + pnl_pct / 100)}


def mark_to_market(pos: dict | None, equity: float, close: float) -> float:
    """Equity do snapshot deste candle (mesma conta da eq_curve do backtest)."""
    if pos is None:
        return equity
    return equity * (1 + pos["side"] * (close / pos["entry_price"] - 1) * pos["exposure"])


def close_position_now(dep: dict, pos: dict) -> None:
    """Fecha a posição paper no close do último candle FECHADO (taker).
    Usado pelo stop manual (api) e pelos guardrails do runner."""
    from providers.market_data import fetch_ohlcv
    import pandas as pd
    from . import store

    df = fetch_ohlcv(dep["symbol"], dep["interval"], exchange=dep["exchange"],
                     limit=5)
    # penúltima linha: a última pode estar em formação
    close = float(df["Close"].iloc[-2])
    ts = int((df.index[-2] - pd.Timestamp(0)).total_seconds() * 1000)
    res = close_pnl(pos, close, TAKER, float(dep["equity"]))
    store.update_position(pos["id"], status="closed", exit_price=close,
                          exit_candle_ts=ts, exit_reason="Fechado manualmente",
                          pnl_pct=res["pnl_pct"], pnl_quote=res["pnl_quote"],
                          fees_quote=res["fees_quote"])
    store.update_deployment(dep["id"], equity=res["new_equity"])
