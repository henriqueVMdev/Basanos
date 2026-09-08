"""
Executor Bybit DEMO — ordens reais no matching da exchange, saldo virtual.

Setup: chaves em BYBIT_DEMO_API_KEY/SECRET (.env). O cliente ccxt roteia
para api-demo.bybit.com via enable_demo_trading(True).

Desenho (por tick do runner, dirigido pelos candles FECHADOS):
1. RECONCILIAR antes de decidir: fetch_positions + fetch_open_orders vs DB.
   - DB aberto / exchange flat  -> TP/SL server-side fechou: resolve o PnL
     real via fetch_my_trades e fecha no DB.
   - exchange aberto / DB flat  -> adota a posição (restart no meio do trade).
   - ordem órfã na exchange     -> cancela.
2. Para cada candle fechado novo: envelhece a posição (bars_held) e aplica
   time-stop; se o runner esteve offline e o prazo venceu, fecha a mercado
   já ('time_stop_late') — o risco seguiu limitado pelo SL server-side.
3. Flat: signal() -> ordem limite PostOnly com TP (limite) e SL (mercado)
   ANEXADOS (tpslMode Full) — ficam na exchange e sobrevivem a PC desligado.
   Ordem antiga não preenchida: amend do preço (novo sinal) ou cancel.
4. Equity do deployment = initial_capital + Σ pnl_quote dos PRÓPRIOS trades
   (o saldo demo é compartilhado entre deployments — fetch_balance mentiria).

Anti-crash: intenção gravada no DB antes da chamada à exchange; ordem sem
exchange_order_id é re-sincronizada contra fetch_open_orders no reconcile.
"""

from __future__ import annotations

import os

import pandas as pd

from . import store

# Perfis de credenciais: None = demo trading; 'personal'/'prop' = mainnet
# (dinheiro REAL) com as chaves do prefixo correspondente no .env.
_ENV_PREFIX = {None: "BYBIT_DEMO", "personal": "BYBIT_REAL", "prop": "BYBIT_PROP"}
_clients: dict = {}


def profile_for(dep) -> str | None:
    """Perfil de chaves do deployment: conta real escolhida ou demo."""
    return dep.get("account") if dep.get("mode") == "real" else None


def _get_client(profile=None):
    if profile in _clients:
        return _clients[profile]
    if profile not in _ENV_PREFIX:
        raise RuntimeError(f"perfil de conta desconhecido: {profile!r}")
    import ccxt
    from dotenv import load_dotenv
    load_dotenv()
    prefix = _ENV_PREFIX[profile]
    key = os.getenv(f"{prefix}_API_KEY", "").strip()
    secret = os.getenv(f"{prefix}_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError(
            f"{prefix}_API_KEY/SECRET ausentes no .env"
            + (" — crie a chave na seção Demo Trading da Bybit (gratuita)"
               if profile is None else ""))
    ex = ccxt.bybit({
        "apiKey": key, "secret": secret,
        "enableRateLimit": True, "timeout": 30000,
        "options": {"defaultType": "swap"},
    })
    if profile is None:
        ex.enable_demo_trading(True)      # mainnet (dinheiro real) nos demais
    ex.load_markets()
    _clients[profile] = ex
    return ex


def _sym(dep) -> str:
    from providers.market_data import normalize_symbol
    return normalize_symbol(dep["symbol"], "bybit")


def _tf_ms(interval: str) -> int:
    from .runner import tf_ms
    return tf_ms(interval)


# ── Reconciliação ────────────────────────────────────────────────────────

def _resolve_closed_position(ex, sym, dep, pos):
    """Posição sumiu da exchange: TP/SL server-side executou. Busca os fills
    reais para PnL/fees e fecha no DB."""
    exit_price, fees = None, 0.0
    try:
        trades = ex.fetch_my_trades(sym, since=int(pos["entry_candle_ts"]), limit=100)
        closing_side = "sell" if pos["side"] == 1 else "buy"
        closes = [t for t in trades if t.get("side") == closing_side]
        if closes:
            qty_sum = sum(float(t["amount"]) for t in closes)
            exit_price = (sum(float(t["price"]) * float(t["amount"]) for t in closes)
                          / qty_sum) if qty_sum else None
            fees = sum(float(t["fee"]["cost"]) for t in closes
                       if t.get("fee") and t["fee"].get("cost") is not None)
    except Exception:
        pass

    if exit_price is None:                      # fallback: usa TP/SL teórico
        exit_price = pos["tp_price"] if pos["side"] == 1 else pos["sl_price"]

    # motivo pelo preço mais próximo (TP vs SL)
    d_tp = abs(exit_price - (pos["tp_price"] or exit_price))
    d_sl = abs(exit_price - (pos["sl_price"] or exit_price))
    reason = "Alvo (maker)" if d_tp <= d_sl else "Stop Loss"

    gross_pct = (pos["side"] * (exit_price / pos["entry_price"] - 1)
                 * 100 * pos["exposure"])
    equity = float(dep["equity"])
    pnl_quote = equity * gross_pct / 100 - fees
    pnl_pct = pnl_quote / equity * 100 if equity else 0.0
    new_equity = equity + pnl_quote

    store.update_position(pos["id"], status="closed", exit_price=exit_price,
                          exit_candle_ts=None, exit_reason=reason,
                          pnl_pct=pnl_pct, pnl_quote=pnl_quote, fees_quote=fees)
    store.update_deployment(dep["id"], equity=new_equity)
    store.add_event(dep["id"], "position_closed",
                    f"{reason} (server-side) @ {exit_price:.2f} ({pnl_pct:+.2f}%)")
    dep["equity"] = new_equity


def _reconcile(ex, sym, dep):
    """Alinha DB <-> exchange. Retorna (pos_db, open_orders_da_exchange)."""
    dep_id = dep["id"]
    ex_positions = [p for p in ex.fetch_positions([sym])
                    if abs(float(p.get("contracts") or 0)) > 0]
    ex_orders = ex.fetch_open_orders(sym)
    pos_db = store.get_open_position(dep_id)

    if pos_db and not ex_positions:
        _resolve_closed_position(ex, sym, dep, pos_db)
        pos_db = None

    if ex_positions and not pos_db:
        # restart no meio do trade: adota a posição real
        p = ex_positions[0]
        side = 1 if p.get("side") == "long" else -1
        entry = float(p.get("entryPrice") or 0)
        params = dep["params"]
        tp_pct = float(params.get("tp_pct", 0.5))
        sl_pct = float(params.get("sl_pct", 1.5))
        pos_id = store.open_position(
            dep_id, side, float(p.get("contracts") or 0),
            min(float(params.get("risk_per_trade", 1.0)) / sl_pct,
                float(params.get("leverage", 2.0))),
            entry, int(dep["last_candle_ts"] or 0),
            entry * (1 + tp_pct / 100 * side), entry * (1 - sl_pct / 100 * side),
            int(params.get("max_bars", 48)))
        store.add_event(dep_id, "position_adopted",
                        f"Posição adotada da exchange @ {entry:.2f}", level="warn")
        pos_db = store.get_open_position(dep_id)

    # ordens: casa DB <-> exchange
    order_db = store.get_working_order(dep_id)
    ex_ids = {o["id"] for o in ex_orders}
    if order_db:
        oid = order_db.get("exchange_order_id")
        if oid and oid not in ex_ids:
            # sumiu da exchange: preenchida ou cancelada
            try:
                o = ex.fetch_order(oid, sym)
                status = o.get("status")
            except Exception:
                status = "canceled"
            if status == "closed":
                store.update_order(order_db["id"], status="filled")
                store.add_event(dep_id, "order_filled",
                                f"Entrada @ {order_db['price']:.2f} (exchange)")
            else:
                store.update_order(order_db["id"], status="cancelled")
            order_db = None
        elif not oid:
            # intenção gravada mas chamada à exchange pode ter falhado
            match = [o for o in ex_orders
                     if abs(float(o.get("price") or 0) - order_db["price"]) < 1e-6]
            if match:
                store.update_order(order_db["id"], exchange_order_id=match[0]["id"])
            else:
                store.update_order(order_db["id"], status="rejected")
                order_db = None

    # ordem órfã (na exchange, sem registro nosso): cancela por segurança
    known = {store.get_working_order(dep_id) or {}}
    known_ids = {o.get("exchange_order_id") for o in known if o}
    for o in ex_orders:
        if o["id"] not in known_ids:
            try:
                ex.cancel_order(o["id"], sym)
                store.add_event(dep_id, "orphan_cancelled",
                                f"Ordem órfã {o['id']} cancelada", level="warn")
            except Exception:
                pass

    return pos_db, ex_orders


# ── Ciclo principal ──────────────────────────────────────────────────────

def process(dep, df_closed, interval):
    """Chamado pelo runner a cada tick para deployments mode='demo'|'real'."""
    from . import signals
    from . import engine_paper as engine

    ex = _get_client(profile_for(dep))
    sym = _sym(dep)
    dep_id = dep["id"]
    tfm = _tf_ms(interval)

    ts_all = ((df_closed.index - pd.Timestamp(0)).total_seconds().values
              * 1000).astype("int64")
    last = dep["last_candle_ts"]
    if last is None:
        last = int(ts_all[-2]) if len(ts_all) >= 2 else int(ts_all[-1]) - tfm

    pos, _ = _reconcile(ex, sym, dep)

    new_idx = [k for k, t in enumerate(ts_all) if t > last]
    for k in new_idx:
        candle_ts = int(ts_all[k])
        close = float(df_closed["Close"].iloc[k])

        if pos:
            bars = pos["bars_held"] + 1
            store.update_position(pos["id"], bars_held=bars)
            pos["bars_held"] = bars
            if bars >= (pos["max_bars"] or 48):
                _close_market(ex, sym, dep, pos,
                              late=(candle_ts < int(ts_all[-1])))
                pos = None

        # snapshot por candle (mark-to-market com o close)
        store.add_equity_snapshot(
            dep_id, candle_ts,
            engine.mark_to_market(pos, float(dep["equity"]), close))
        store.update_deployment(dep_id, last_candle_ts=candle_ts)

    # decide a ordem de entrada com o último candle fechado
    if pos is None and new_idx:
        sig = None
        if len(df_closed) >= 400:
            sig = signals.get_signal(dep["strategy_file"], df_closed, dep["params"])
        order_db = store.get_working_order(dep_id)
        if sig:
            _place_or_amend(ex, sym, dep, order_db, sig, int(ts_all[-1]) + tfm)
        elif order_db and order_db.get("exchange_order_id"):
            try:
                ex.cancel_order(order_db["exchange_order_id"], sym)
            except Exception:
                pass
            store.update_order(order_db["id"], status="cancelled")
            store.add_event(dep_id, "order_cancelled", "Sem setup — limite cancelada")


def _place_or_amend(ex, sym, dep, order_db, sig, valid_ts):
    from . import engine_paper  # noqa: F401 (fees ficam na exchange)
    dep_id = dep["id"]
    price = float(ex.price_to_precision(sym, sig["price"]))
    side_str = "buy" if sig["side"] == 1 else "sell"
    equity = float(dep["equity"])
    qty_raw = sig["exposure"] * equity / price if price > 0 else 0
    qty = float(ex.amount_to_precision(sym, qty_raw))

    # Proteções de sizing OPCIONAIS (default: desligadas — sizing puro da
    # estratégia sobre o capital virtual do deployment)
    g = dep.get("guardrails") or {}
    if g.get("max_notional") and qty * price > float(g["max_notional"]):
        qty = float(ex.amount_to_precision(sym, float(g["max_notional"]) / price))
        store.add_event(dep_id, "order_capped",
                        f"Notional limitado a {g['max_notional']} (proteção)", level="warn")
    if dep.get("mode") == "real" and g.get("check_balance"):
        try:
            free = float(((ex.fetch_balance().get("USDT") or {}).get("free")) or 0)
        except Exception as e:
            store.add_event(dep_id, "order_skipped",
                            f"check_balance falhou: {str(e)[:120]}", level="warn")
            return
        if free < qty * price:
            store.add_event(dep_id, "order_skipped",
                            f"saldo USDT livre {free:.2f} < notional {qty * price:.2f}",
                            level="warn")
            return

    mkt = ex.market(sym)
    min_qty = ((mkt.get("limits") or {}).get("amount") or {}).get("min") or 0
    if qty < min_qty:
        store.add_event(dep_id, "order_skipped",
                        f"qty {qty} < mínimo {min_qty} da exchange", level="warn")
        return

    tp = float(ex.price_to_precision(sym, price * (1 + sig["tp_pct"] / 100 * sig["side"])))
    sl = float(ex.price_to_precision(sym, price * (1 - sig["sl_pct"] / 100 * sig["side"])))

    if order_db and order_db.get("exchange_order_id"):
        # amend do preço (1 chamada); TP/SL anexados acompanham a ordem
        try:
            ex.edit_order(order_db["exchange_order_id"], sym, "limit", side_str,
                          qty, price,
                          params={"takeProfit": tp, "stopLoss": sl})
            store.update_order(order_db["id"], price=price, qty=qty,
                               valid_candle_ts=valid_ts, side=sig["side"])
            store.add_event(dep_id, "order_amended",
                            f"Limite movida p/ {price:.2f}")
            return
        except Exception:
            try:
                ex.cancel_order(order_db["exchange_order_id"], sym)
            except Exception:
                pass
            store.update_order(order_db["id"], status="cancelled")

    # intenção ANTES da chamada (anti-crash), id depois
    oid_db = store.create_order(dep_id, "entry", sig["side"], "limit", price,
                                qty, valid_ts, tp_pct=sig["tp_pct"],
                                sl_pct=sig["sl_pct"], max_bars=sig["max_bars"],
                                exposure=sig["exposure"])
    try:
        o = ex.create_order(sym, "limit", side_str, qty, price, params={
            "timeInForce": "PO",              # PostOnly: garante maker
            "takeProfit": tp, "stopLoss": sl,
            "tpslMode": "Full",
            "tpOrderType": "Limit", "tpLimitPrice": tp,
            "slOrderType": "Market",
            "positionIdx": 0,
        })
        store.update_order(oid_db, exchange_order_id=o["id"])
        store.add_event(dep_id, "order_placed",
                        f"Limite {side_str} @ {price:.2f} (TP {tp:.2f} / SL {sl:.2f} "
                        "anexados server-side)")
    except Exception as e:
        store.update_order(oid_db, status="rejected")
        store.add_event(dep_id, "order_rejected", str(e)[:300], level="warn")


def stop_close(dep, pos):
    """Parada definitiva (stop manual ou guardrail): cancela a ordem pendente
    na exchange, fecha a posição a mercado e reconcilia JÁ — o deployment
    parado não ticka mais, então o PnL real precisa ser resolvido agora."""
    ex = _get_client(profile_for(dep))
    sym = _sym(dep)
    order = store.get_working_order(dep["id"])
    if order:
        if order.get("exchange_order_id"):
            try:
                ex.cancel_order(order["exchange_order_id"], sym)
            except Exception:
                pass
        store.update_order(order["id"], status="cancelled")
    if pos:
        import time as _time
        side_str = "sell" if pos["side"] == 1 else "buy"
        ex.create_order(sym, "market", side_str, pos["qty"],
                        params={"reduceOnly": True, "positionIdx": 0})
        # resolve PnL/fees reais da saída; 1 retry se o fill ainda não refletiu
        for _ in range(2):
            _time.sleep(1.5)
            _reconcile(ex, sym, dep)
            if store.get_open_position(dep["id"]) is None:
                break


def _close_market(ex, sym, dep, pos, late=False):
    dep_id = dep["id"]
    side_str = "sell" if pos["side"] == 1 else "buy"
    try:
        ex.create_order(sym, "market", side_str, pos["qty"],
                        params={"reduceOnly": True, "positionIdx": 0})
    except Exception as e:
        store.add_event(dep_id, "close_failed", str(e)[:300], level="error")
        return
    reason = "Time Stop (atrasado)" if late else "Time Stop"
    store.add_event(dep_id, "time_stop", f"{reason} — fechamento a mercado enviado",
                    level="warn" if late else "info")
    # o PnL real entra no próximo reconcile (posição some da exchange)
    pos["max_bars"] = 10 ** 9          # evita reenvio no mesmo tick
