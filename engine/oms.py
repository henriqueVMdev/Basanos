"""
OMS/EMS — execução de ordens multi-ativo.

Contas:
- paper       : motor simulado local, multi-classe (cripto, ações/ETFs — incl.
                ETFs de renda fixa —, câmbio e futuros). Fills a mercado usam
                bid/ask (cripto: VWAP andando no book L2; tradfi: top-of-book
                do Yahoo, dados atrasados ~15min — SEMPRE rotulado na UI).
                Ordens limite ficam WORKING e são checadas a cada poll do
                blotter (fill no preço limite quando o mercado cruza, fee maker).
- bybit_demo / bybit_real / bybit_prop : ordens REAIS na Bybit (perps USDT)
                via ccxt, chaves no .env. Somente market='crypto'. Conta real
                exige confirm=true no payload.

Pré-trade: notional, fees estimadas, spread, slippage estimada (book L2 p/
cripto; heurística spread/2 + impacto √(%ADV) p/ tradfi), %ADV, vol diária,
VaR 1d 95%. Pós-trade (TCA): slippage vs. mid de chegada por fill,
implementation shortfall, agregados por mercado.

Persistência: oms_data.json (mesmo padrão do alerts_data.json).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

_FILE = Path(__file__).parents[1] / "data" / "oms_data.json"
_lock = threading.Lock()

from common import ttl_cache, to_float

_cached = ttl_cache()


# Persistência
_DEFAULT_STATE = {
    "config": {"paper_capital": 100_000.0, "paper_cash": 100_000.0},
    "orders": [],
    "fills": [],
    "positions": {},          # {account: {"SYM|market": {...}}}
}


def _load() -> dict:
    if not _FILE.exists():
        return json.loads(json.dumps(_DEFAULT_STATE))
    try:
        st = json.loads(_FILE.read_text(encoding="utf-8"))
        for k, v in _DEFAULT_STATE.items():
            st.setdefault(k, json.loads(json.dumps(v)))
        return st
    except (ValueError, OSError):
        return json.loads(json.dumps(_DEFAULT_STATE))


def _save(st: dict) -> None:
    _FILE.write_text(json.dumps(st, ensure_ascii=False, indent=1),
                     encoding="utf-8")


# Contas
_BYBIT_PREFIX = {"bybit_demo": "BYBIT_DEMO", "bybit_real": "BYBIT_REAL",
                 "bybit_prop": "BYBIT_PROP"}
_bybit_clients: dict = {}

# taxas Bybit perps (taker/maker); tradfi: corretoras US zero-commission —
# o custo real é spread+slippage, que o pré-trade estima à parte
_FEES = {"crypto": {"taker": 0.00055, "maker": 0.0002},
         "tradfi": {"taker": 0.0, "maker": 0.0}}


def accounts() -> dict:
    st = _load()
    out = [{
        "id": "paper", "label": "Paper (simulado)", "kind": "paper",
        "configured": True, "markets": ["crypto", "tradfi"],
        "cash": st["config"].get("paper_cash"),
        "capital": st["config"].get("paper_capital"),
        "note": "fills simulados sobre dados de mercado; tradfi atrasado ~15min",
    }]
    for acc, prefix in _BYBIT_PREFIX.items():
        configured = bool(os.getenv(f"{prefix}_API_KEY", "").strip())
        if not configured:
            from dotenv import load_dotenv
            load_dotenv()
            configured = bool(os.getenv(f"{prefix}_API_KEY", "").strip())
        out.append({
            "id": acc,
            "label": {"bybit_demo": "Bybit DEMO (saldo virtual)",
                      "bybit_real": "Bybit REAL — dinheiro real",
                      "bybit_prop": "Bybit PROP — dinheiro real"}[acc],
            "kind": "demo" if acc == "bybit_demo" else "real",
            "configured": configured, "markets": ["crypto"],
            "note": None if configured
            else f"chaves {prefix}_API_KEY/SECRET ausentes no .env",
        })
    return {"accounts": out}


def _bybit_client(account: str):
    if account in _bybit_clients:
        return _bybit_clients[account]
    prefix = _BYBIT_PREFIX[account]
    import ccxt
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv(f"{prefix}_API_KEY", "").strip()
    secret = os.getenv(f"{prefix}_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError(f"{prefix}_API_KEY/SECRET ausentes no .env")
    ex = ccxt.bybit({"apiKey": key, "secret": secret,
                     "enableRateLimit": True, "timeout": 30000,
                     "options": {"defaultType": "swap"}})
    if account == "bybit_demo":
        ex.enable_demo_trading(True)
    ex.load_markets()
    _bybit_clients[account] = ex
    return ex


# Cotação/mark por classe
# meia-spread estimada (bps) quando o Yahoo não expõe bid/ask (ex.: fora do
# pregão, futuros, FX) — usada nos fills paper e SEMPRE sinalizada
_HALF_SPREAD_BPS = {"equity": 2.0, "fx": 1.0, "future": 3.0}


def _asset_class(yf_sym: str) -> str:
    if yf_sym.endswith("=X"):
        return "fx"
    if "=F" in yf_sym or yf_sym.endswith((".NYM", ".NYB", ".CBT", ".CME", ".CMX")):
        return "future"
    return "equity"


def mark(symbol: str, market: str, exchange: str = "bybit") -> dict:
    """bid/ask/mid/last atuais. Cripto: ticker ccxt (5s). Tradfi: Yahoo (15s,
    atrasado ~15min; sem bid/ask usa last ± meia-spread estimada por classe)."""
    if market == "crypto":
        from providers.market_data import get_exchange, normalize_symbol
        pair = normalize_symbol(symbol, exchange)

        def fetch():
            t = get_exchange(exchange).fetch_ticker(pair)
            bid, ask = to_float(t.get("bid")), to_float(t.get("ask"))
            last = to_float(t.get("last"))
            return {"market": "crypto", "resolved": pair,
                    "bid": bid, "ask": ask, "last": last,
                    "mid": (bid + ask) / 2 if bid and ask else last,
                    "adv_usd": to_float(t.get("quoteVolume")),
                    "delayed": False, "spread_estimated": False}

        return _cached(("mark", "c", pair), 5, fetch)

    from providers import tradfi_data
    yf_sym = tradfi_data.resolve(symbol)
    if yf_sym.startswith("^"):
        raise ValueError(f"{yf_sym} é um índice — não negociável; "
                         "use o futuro ou um ETF do índice")

    def fetch():
        import yfinance as yf
        t = yf.Ticker(yf_sym)
        fi = t.fast_info
        last = to_float(fi["last_price"])
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass
        bid, ask = to_float(info.get("bid")), to_float(info.get("ask"))
        estimated = False
        stale = (not bid or not ask or bid >= ask          # cruzado = dado velho
                 or (last and (abs(bid / last - 1) > 0.05
                               or abs(ask / last - 1) > 0.05)))
        if stale:
            half = _HALF_SPREAD_BPS[_asset_class(yf_sym)] / 1e4
            bid, ask = last * (1 - half), last * (1 + half)
            estimated = True
        adv_units = to_float(info.get("averageVolume")) or to_float(fi["last_volume"])
        return {"market": "tradfi", "resolved": yf_sym,
                "bid": bid, "ask": ask, "last": last,
                "mid": (bid + ask) / 2 if bid and ask else last,
                "adv_usd": adv_units * last if adv_units and last else None,
                "delayed": True, "spread_estimated": estimated}

    return _cached(("mark", "t", yf_sym), 15, fetch)


def _daily_vol(symbol: str, market: str, exchange: str) -> float | None:
    def fetch():
        from providers import technical_data
        df = technical_data.history(symbol, market, "1d", 60, exchange)
        if df.empty or len(df) < 10:
            return None
        return float(df["Close"].pct_change().dropna().std())

    try:
        return _cached(("dvol", market, symbol.upper()), 1800, fetch)
    except Exception:
        return None


def _walk_book(levels: list, qty: float):
    """VWAP p/ executar qty sobre níveis [[preço, tamanho], ...] do book."""
    remaining, cost = qty, 0.0
    for price, size in levels:
        take = min(remaining, size)
        cost += take * price
        remaining -= take
        if remaining <= 1e-12:
            return cost / qty, qty
    filled = qty - remaining
    return (cost / filled if filled else None), filled


# Pré-trade analytics
def pre_trade(body: dict) -> dict:
    symbol = (body.get("symbol") or "").strip().upper()
    market = (body.get("market") or "crypto").lower()
    side = (body.get("side") or "buy").lower()
    qty = to_float(body.get("qty"))
    exchange = (body.get("exchange") or "bybit").lower()
    order_type = (body.get("type") or "market").lower()
    if not symbol or not qty or qty <= 0:
        raise ValueError("symbol e qty (> 0) obrigatórios")

    m = mark(symbol, market, exchange)
    mid = m["mid"]
    if not mid:
        raise ValueError(f"sem cotação para {symbol}")
    notional = qty * mid
    spread_bps = ((m["ask"] - m["bid"]) / mid * 1e4
                  if m["bid"] and m["ask"] else None)

    warnings = []
    slippage_bps, book_coverage = None, None
    if market == "crypto":
        from providers.market_data import get_exchange, normalize_symbol
        pair = normalize_symbol(symbol, exchange)
        ob = get_exchange(exchange).fetch_order_book(pair, limit=50)
        levels = ob["asks"] if side == "buy" else ob["bids"]
        vwap, filled = _walk_book([lv[:2] for lv in levels], qty)
        book_coverage = filled / qty if qty else None
        if vwap:
            slippage_bps = abs(vwap - mid) / mid * 1e4
        if book_coverage is not None and book_coverage < 1:
            warnings.append(f"book L2 (50 níveis) cobre só "
                            f"{book_coverage * 100:.0f}% da ordem — "
                            "slippage real será maior")
    else:
        # heurística: meia-spread + impacto ~ 10bps × √(% do ADV)
        pct_adv = notional / m["adv_usd"] if m.get("adv_usd") else None
        impact = 10.0 * math.sqrt(pct_adv) if pct_adv else 0.0
        slippage_bps = (spread_bps or 0) / 2 + impact
        warnings.append("estimativa heurística (sem book L2 público p/ tradfi)")

    pct_adv = notional / m["adv_usd"] if m.get("adv_usd") else None
    fee_rate = _FEES[market]["maker" if order_type == "limit" else "taker"]
    fee_est = notional * fee_rate
    vol_d = _daily_vol(symbol, market, exchange)
    var_1d = 1.65 * vol_d * notional if vol_d else None

    if m.get("delayed"):
        warnings.append("dados atrasados ~15min (Yahoo) — fills paper usam esse preço")
    if m.get("spread_estimated"):
        warnings.append("bid/ask indisponível — spread ESTIMADO por classe de ativo")
    if pct_adv and pct_adv > 0.01:
        warnings.append(f"ordem = {pct_adv * 100:.1f}% do volume diário — "
                        "impacto de mercado relevante")
    if spread_bps and spread_bps > 25:
        warnings.append(f"spread largo ({spread_bps:.0f} bps)")
    if market == "tradfi" and _FEES["tradfi"]["taker"] == 0:
        warnings.append("fee 0 assume corretora zero-commission; custo real = spread+slippage")

    cost_bps = (fee_rate * 1e4) + (slippage_bps or 0)
    return {
        "symbol": symbol, "market": market, "side": side, "qty": qty,
        "resolved": m["resolved"], "bid": m["bid"], "ask": m["ask"],
        "mid": mid, "last": m["last"], "notional": notional,
        "spread_bps": spread_bps, "slippage_est_bps": slippage_bps,
        "book_coverage": book_coverage,
        "fee_rate_bps": fee_rate * 1e4, "fee_est": fee_est,
        "cost_total_bps": cost_bps, "cost_total_usd": notional * cost_bps / 1e4,
        "pct_adv": pct_adv, "adv_usd": m.get("adv_usd"),
        "daily_vol_pct": vol_d * 100 if vol_d else None,
        "var_1d_95": var_1d, "delayed": m.get("delayed"),
        "warnings": warnings,
    }


# Ledger de posições
def _pos_key(symbol: str, market: str) -> str:
    return f"{symbol.upper()}|{market}"


def _apply_fill(st: dict, account: str, symbol: str, market: str,
                side: str, qty: float, price: float, fee: float) -> None:
    key = _pos_key(symbol, market)
    pos = st["positions"].setdefault(account, {}).setdefault(key, {
        "symbol": symbol.upper(), "market": market,
        "qty": 0.0, "avg_price": 0.0, "realized": 0.0, "fees": 0.0,
    })
    signed = qty if side == "buy" else -qty
    old = pos["qty"]
    if old * signed >= 0:                                   # abre/aumenta
        new_qty = old + signed
        if new_qty:
            pos["avg_price"] = ((abs(old) * pos["avg_price"] + qty * price)
                                / abs(new_qty))
        pos["qty"] = new_qty
    else:                                                    # reduz/inverte
        closing = min(qty, abs(old))
        direction = 1 if old > 0 else -1
        pos["realized"] += closing * (price - pos["avg_price"]) * direction
        new_qty = old + signed
        if old * new_qty < 0:                                # inverteu o lado
            pos["avg_price"] = price
        elif new_qty == 0:
            pos["avg_price"] = 0.0
        pos["qty"] = new_qty
    pos["fees"] += fee
    if account == "paper":
        cash = st["config"].get("paper_cash", 0.0)
        st["config"]["paper_cash"] = (cash - qty * price - fee if side == "buy"
                                      else cash + qty * price - fee)


def _record_fill(st: dict, order: dict, qty: float, price: float,
                 liquidity: str) -> None:
    fee = qty * price * _FEES[order["market"]][liquidity]
    arrival_mid = (order.get("arrival") or {}).get("mid")
    sgn = 1 if order["side"] == "buy" else -1
    slippage = (sgn * (price - arrival_mid) / arrival_mid * 1e4
                if arrival_mid else None)
    now = int(time.time() * 1000)
    st["fills"].append({
        "id": uuid.uuid4().hex[:10], "order_id": order["id"], "ts": now,
        "account": order["account"], "market": order["market"],
        "symbol": order["symbol"], "side": order["side"],
        "qty": qty, "price": price, "fee": fee, "liquidity": liquidity,
        "arrival_mid": arrival_mid, "slippage_bps": slippage,
        "latency_ms": now - order["ts"],
    })
    order["filled_qty"] = (order.get("filled_qty") or 0) + qty
    prev_avg = order.get("avg_price") or 0.0
    prev_qty = order["filled_qty"] - qty
    order["avg_price"] = ((prev_avg * prev_qty + price * qty)
                          / order["filled_qty"])
    order["fee"] = (order.get("fee") or 0) + fee
    order["status"] = ("filled" if order["filled_qty"] >= order["qty"] - 1e-12
                       else "partial")
    _apply_fill(st, order["account"], order["symbol"], order["market"],
                order["side"], qty, price, fee)


# Envio de ordens
def submit_order(body: dict) -> dict:
    account = (body.get("account") or "paper").lower()
    symbol = (body.get("symbol") or "").strip().upper()
    market = (body.get("market") or "crypto").lower()
    side = (body.get("side") or "").lower()
    order_type = (body.get("type") or "market").lower()
    qty = to_float(body.get("qty"))
    limit_price = to_float(body.get("limit_price"))
    exchange = (body.get("exchange") or "bybit").lower()

    if side not in ("buy", "sell"):
        raise ValueError("side deve ser buy ou sell")
    if order_type not in ("market", "limit"):
        raise ValueError("type deve ser market ou limit")
    if not symbol or not qty or qty <= 0:
        raise ValueError("symbol e qty (> 0) obrigatórios")
    if order_type == "limit" and not limit_price:
        raise ValueError("limit_price obrigatório em ordem limite")
    if account in _BYBIT_PREFIX and market != "crypto":
        raise ValueError("contas Bybit só executam cripto — "
                         "use a conta paper p/ tradfi")
    if account in ("bybit_real", "bybit_prop") and not body.get("confirm"):
        raise ValueError("ordem em conta REAL exige confirm=true")
    if account not in ("paper", *_BYBIT_PREFIX):
        raise ValueError(f"conta desconhecida: {account}")

    m = mark(symbol, market, exchange)          # valida símbolo + arrival p/ TCA
    order = {
        "id": uuid.uuid4().hex[:10], "ts": int(time.time() * 1000),
        "account": account, "market": market, "symbol": symbol,
        "resolved": m["resolved"], "exchange": exchange if market == "crypto" else None,
        "side": side, "type": order_type, "qty": qty,
        "limit_price": limit_price, "status": "working",
        "arrival": {"bid": m["bid"], "ask": m["ask"], "mid": m["mid"],
                    "last": m["last"], "ts": int(time.time() * 1000)},
        "filled_qty": 0.0, "avg_price": None, "fee": 0.0,
        "exchange_order_id": None, "delayed_data": bool(m.get("delayed")),
    }

    if account == "paper":
        with _lock:
            st = _load()
            if order_type == "market":
                price = None
                if market == "crypto":
                    from providers.market_data import get_exchange, normalize_symbol
                    pair = normalize_symbol(symbol, exchange)
                    ob = get_exchange(exchange).fetch_order_book(pair, limit=50)
                    levels = ob["asks"] if side == "buy" else ob["bids"]
                    price, filled = _walk_book([lv[:2] for lv in levels], qty)
                    if price is None or filled < qty:
                        order["status"] = "rejected"
                        order["note"] = "book L2 raso demais para a quantidade"
                if price is None and order["status"] != "rejected":
                    price = m["ask"] if side == "buy" else m["bid"]
                if order["status"] != "rejected":
                    _record_fill(st, order, qty, float(price), "taker")
            st["orders"].append(order)
            _save(st)
        return order

    # contas Bybit: ordem REAL via ccxt
    ex = _bybit_client(account)
    from providers.market_data import normalize_symbol
    pair = normalize_symbol(symbol, "bybit")
    qty_p = float(ex.amount_to_precision(pair, qty))
    try:
        if order_type == "market":
            res = ex.create_order(pair, "market", side, qty_p,
                                  params={"positionIdx": 0})
        else:
            price_p = float(ex.price_to_precision(pair, limit_price))
            res = ex.create_order(pair, "limit", side, qty_p, price_p,
                                  params={"positionIdx": 0})
        order["exchange_order_id"] = res.get("id")
        order["resolved"] = pair
    except Exception as e:
        order["status"] = "rejected"
        order["note"] = str(e)[:300]
    with _lock:
        st = _load()
        st["orders"].append(order)
        _save(st)
    if order["exchange_order_id"]:
        _sync_bybit_orders(account)             # pode já ter preenchido (market)
    with _lock:
        st = _load()
        cur = next((o for o in st["orders"] if o["id"] == order["id"]), order)
    return cur


def cancel_order(order_id: str) -> dict:
    with _lock:
        st = _load()
        order = next((o for o in st["orders"] if o["id"] == order_id), None)
        if not order:
            raise ValueError("ordem não encontrada")
        if order["status"] not in ("working", "partial"):
            raise ValueError(f"ordem já está {order['status']}")
        if order["account"] == "paper":
            order["status"] = "cancelled"
            _save(st)
            return order
    # bybit: cancela na exchange e depois persiste
    ex = _bybit_client(order["account"])
    try:
        ex.cancel_order(order["exchange_order_id"], order["resolved"])
    except Exception as e:
        raise RuntimeError(f"cancel na exchange falhou: {str(e)[:200]}")
    with _lock:
        st = _load()
        o = next((o for o in st["orders"] if o["id"] == order_id), None)
        if o:
            o["status"] = "cancelled"
            _save(st)
        return o or order


# Motor de fills paper (lazy: roda a cada poll do blotter)
def _check_paper_limits(st: dict) -> bool:
    changed = False
    for order in st["orders"]:
        if order["account"] != "paper" or order["status"] != "working":
            continue
        if order["type"] != "limit":
            continue
        try:
            m = mark(order["symbol"], order["market"],
                     order.get("exchange") or "bybit")
        except Exception:
            continue
        lp = order["limit_price"]
        crossed = ((order["side"] == "buy" and m["ask"] and m["ask"] <= lp)
                   or (order["side"] == "sell" and m["bid"] and m["bid"] >= lp))
        if crossed:
            _record_fill(st, order, order["qty"] - order["filled_qty"],
                         lp, "maker")
            changed = True
    return changed


def _sync_bybit_orders(account: str) -> None:
    """Atualiza status/fills das ordens Bybit não finais desta conta."""
    with _lock:
        st = _load()
        open_orders = [o for o in st["orders"]
                       if o["account"] == account and o["exchange_order_id"]
                       and o["status"] in ("working", "partial")]
    if not open_orders:
        return
    ex = _bybit_client(account)
    updates = []
    for o in open_orders:
        try:
            res = ex.fetch_order(o["exchange_order_id"], o["resolved"])
        except Exception:
            continue
        updates.append((o["id"], res))
    if not updates:
        return
    with _lock:
        st = _load()
        by_id = {o["id"]: o for o in st["orders"]}
        changed = False
        for oid, res in updates:
            o = by_id.get(oid)
            if not o or o["status"] not in ("working", "partial"):
                continue
            filled = to_float(res.get("filled")) or 0.0
            avg = to_float(res.get("average")) or to_float(res.get("price"))
            new_fill = filled - (o.get("filled_qty") or 0)
            if new_fill > 1e-12 and avg:
                _record_fill(st, o, new_fill, avg,
                             "maker" if o["type"] == "limit" else "taker")
                changed = True
            status = res.get("status")
            if status == "canceled" and o["status"] in ("working", "partial"):
                o["status"] = "cancelled"
                changed = True
        if changed:
            _save(st)


# Blotter / monitoramento
def blotter(account: str = "paper") -> dict:
    if account in _BYBIT_PREFIX:
        try:
            _sync_bybit_orders(account)
        except Exception:
            pass
    with _lock:
        st = _load()
        if _check_paper_limits(st):
            _save(st)
        orders = [o for o in st["orders"] if o["account"] == account]
        fills = [f for f in st["fills"] if f["account"] == account]
        positions = dict(st["positions"].get(account) or {})
        cash = st["config"].get("paper_cash") if account == "paper" else None

    # mark-to-market das posições abertas
    pos_rows, unrealized_total = [], 0.0
    for key, p in positions.items():
        if abs(p["qty"]) < 1e-12 and abs(p["realized"]) < 1e-9:
            continue
        mk = None
        if abs(p["qty"]) > 1e-12:
            try:
                mk = mark(p["symbol"], p["market"])
            except Exception:
                mk = None
        mark_px = mk["mid"] if mk else None
        unreal = ((mark_px - p["avg_price"]) * p["qty"]
                  if mark_px and abs(p["qty"]) > 1e-12 else 0.0)
        unrealized_total += unreal or 0.0
        pos_rows.append({**p, "mark": mark_px, "unrealized": unreal,
                         "notional": abs(p["qty"]) * mark_px if mark_px else None,
                         "delayed": mk.get("delayed") if mk else None})

    exchange_positions = []
    if account in _BYBIT_PREFIX:
        try:
            ex = _bybit_client(account)
            for p in ex.fetch_positions():
                contracts = to_float(p.get("contracts"))
                if not contracts:
                    continue
                exchange_positions.append({
                    "symbol": p.get("symbol"), "side": p.get("side"),
                    "qty": contracts, "entry": to_float(p.get("entryPrice")),
                    "mark": to_float(p.get("markPrice")),
                    "unrealized": to_float(p.get("unrealizedPnl")),
                    "leverage": to_float(p.get("leverage")),
                })
        except Exception as e:
            exchange_positions = [{"error": str(e)[:200]}]

    realized_total = sum(p["realized"] for p in positions.values())
    fees_total = sum(p["fees"] for p in positions.values())
    return {
        "account": account,
        "orders": sorted(orders, key=lambda o: -o["ts"])[:200],
        "fills": sorted(fills, key=lambda f: -f["ts"])[:200],
        "positions": sorted(pos_rows, key=lambda p: -(p["notional"] or 0)),
        "exchange_positions": exchange_positions,
        "cash": cash,
        "summary": {"realized": realized_total, "fees": fees_total,
                    "unrealized": unrealized_total,
                    # caixa + valor de mercado assinado das posições (o
                    # não-realizado já está embutido no mark)
                    "equity": ((cash or 0)
                               + sum(p["qty"] * p["mark"] for p in pos_rows
                                     if p.get("mark"))
                               if account == "paper" else None)},
        "ts": int(time.time() * 1000),
    }


# Pós-trade analytics (TCA)
def tca(account: str = "paper") -> dict:
    with _lock:
        st = _load()
        fills = [f for f in st["fills"] if f["account"] == account]
        orders_by_id = {o["id"]: o for o in st["orders"]}

    if not fills:
        return {"account": account, "n_fills": 0, "note": "sem execuções ainda"}

    rows = []
    for f in fills:
        o = orders_by_id.get(f["order_id"]) or {}
        rows.append({**f, "order_type": o.get("type"),
                     "delayed_data": o.get("delayed_data")})

    slips = [f["slippage_bps"] for f in fills if f["slippage_bps"] is not None]
    notional = sum(f["qty"] * f["price"] for f in fills)
    fees = sum(f["fee"] for f in fills)
    shortfall_usd = sum(
        (1 if f["side"] == "buy" else -1) * (f["price"] - f["arrival_mid"])
        * f["qty"] for f in fills if f.get("arrival_mid")) + fees
    slips_sorted = sorted(slips)

    def _by(keyfn):
        groups = {}
        for f in rows:
            groups.setdefault(keyfn(f), []).append(f)
        out = []
        for k, g in groups.items():
            s = [x["slippage_bps"] for x in g if x["slippage_bps"] is not None]
            out.append({"key": k, "n": len(g),
                        "notional": sum(x["qty"] * x["price"] for x in g),
                        "fees": sum(x["fee"] for x in g),
                        "avg_slippage_bps": sum(s) / len(s) if s else None})
        return out

    worst = sorted((f for f in rows if f["slippage_bps"] is not None),
                   key=lambda f: -f["slippage_bps"])[:10]
    return {
        "account": account, "n_fills": len(fills), "notional": notional,
        "fees": fees, "fees_bps": fees / notional * 1e4 if notional else None,
        "avg_slippage_bps": sum(slips) / len(slips) if slips else None,
        "median_slippage_bps": (slips_sorted[len(slips) // 2]
                                if slips else None),
        "implementation_shortfall_usd": shortfall_usd,
        "shortfall_bps": (shortfall_usd / notional * 1e4
                          if notional else None),
        "avg_latency_ms": (sum(f["latency_ms"] for f in fills) / len(fills)),
        "by_market": _by(lambda f: f["market"]),
        "by_type": _by(lambda f: f.get("order_type") or "?"),
        "worst_fills": worst,
        "note": ("slippage de fills paper tradfi é medida sobre dados "
                 "atrasados ~15min — trate como aproximação"
                 if any(f.get("delayed_data") for f in rows) else None),
    }


def reset_paper() -> dict:
    """Zera a conta paper (ordens, fills, posições, caixa). Só paper."""
    with _lock:
        st = _load()
        st["orders"] = [o for o in st["orders"] if o["account"] != "paper"]
        st["fills"] = [f for f in st["fills"] if f["account"] != "paper"]
        st["positions"].pop("paper", None)
        st["config"]["paper_cash"] = st["config"].get("paper_capital", 100_000.0)
        _save(st)
    return {"ok": True, "cash": st["config"]["paper_cash"]}
