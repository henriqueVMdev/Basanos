"""
Blueprint /api/terminal — funções estilo Bloomberg Terminal.

Tudo aditivo: dados públicos via ccxt (market_data.get_exchange, instância
cacheada e rate-limited), caches próprios com TTL, alertas em JSON com lock
(padrão do journal) checados por uma thread daemon iniciada sob demanda.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen

from flask import Blueprint, jsonify, request

from providers.market_data import get_exchange, normalize_symbol

terminal_bp = Blueprint("terminal", __name__, url_prefix="/api/terminal")

# ── cache TTL genérico ───────────────────────────────────────────────────
_CACHE: dict = {}


def _cached(key, ttl_s, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl_s:
        return hit[1]
    data = fn()
    _CACHE[key] = (time.time(), data)
    return data


def _safe_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ── /watch — tickers ao vivo da watchlist ────────────────────────────────

@terminal_bp.get("/watch")
def watch():
    try:
        exchange = (request.args.get("exchange") or "bybit").lower()
        bases = [s.strip() for s in (request.args.get("symbols") or "").split(",")
                 if s.strip()]
        tradfi = [s.strip() for s in (request.args.get("tradfi") or "").split(",")
                  if s.strip()]
        if not bases and not tradfi:
            return jsonify({"rows": []})
        rows = []
        if bases:
            ex = get_exchange(exchange)
            syms = [normalize_symbol(b, exchange) for b in bases]
            tickers = ex.fetch_tickers(syms)
            try:
                frs = ex.fetch_funding_rates(syms)
            except Exception:
                frs = {}
            for base, sym in zip(bases, syms):
                t = tickers.get(sym) or {}
                fr = frs.get(sym) or {}
                rows.append({
                    "base": base.upper(),
                    "symbol": sym,
                    "market": "crypto",
                    "last": _safe_float(t.get("last")),
                    "pct24h": _safe_float(t.get("percentage")),
                    "high24": _safe_float(t.get("high")),
                    "low24": _safe_float(t.get("low")),
                    "vol_usd": _safe_float(t.get("quoteVolume")),
                    "funding": _safe_float(fr.get("fundingRate")),
                    "next_funding_ts": fr.get("fundingTimestamp") or fr.get("nextFundingTimestamp"),
                })
        if tradfi:
            from providers import tradfi_data
            tq = tradfi_data.quotes(tradfi)
            rows.extend(tq[s] for s in tradfi if tq.get(s))
        return jsonify({"rows": rows, "ts": int(time.time() * 1000)})
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── /spark — closes p/ sparkline ─────────────────────────────────────────

@terminal_bp.get("/spark")
def spark():
    try:
        exchange = (request.args.get("exchange") or "bybit").lower()
        base = (request.args.get("symbol") or "").strip()
        tf = request.args.get("tf", "15m")
        bars = min(int(request.args.get("bars", 96)), 500)
        if not base:
            return jsonify({"error": "symbol obrigatório"}), 400
        if (request.args.get("market") or "").lower() == "tradfi":
            from providers import tradfi_data
            return jsonify({"closes": tradfi_data.closes(base, tf, bars)})
        sym = normalize_symbol(base, exchange)

        def fetch():
            ex = get_exchange(exchange)
            raw = ex.fetch_ohlcv(sym, timeframe=tf, limit=bars)
            return [_safe_float(c[4]) for c in raw]

        closes = _cached(("spark", exchange, sym, tf, bars), 600, fetch)
        return jsonify({"closes": closes})
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── /screener — ranking de perps ─────────────────────────────────────────

def _kline_stats(ex, sym):
    """Retornos 1d/7d/30d e ATR14% do símbolo, de klines diárias (cache 30min)."""
    def fetch():
        raw = ex.fetch_ohlcv(sym, timeframe="1d", limit=31)
        if len(raw) < 2:
            return {}
        closes = [c[4] for c in raw]
        highs = [c[2] for c in raw]
        lows = [c[3] for c in raw]
        last = closes[-1]
        def ret(n):
            return (last / closes[-n - 1] - 1) * 100 if len(closes) > n else None
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                   abs(lows[i] - closes[i - 1])) for i in range(1, len(raw))]
        atr = sum(trs[-14:]) / min(14, len(trs))
        return {"ret1d": ret(1), "ret7d": ret(7), "ret30d": ret(30),
                "atr_pct": atr / last * 100 if last else None}
    return _cached(("kstats", sym), 1800, fetch)


@terminal_bp.get("/screener")
def screener():
    try:
        market = (request.args.get("market") or "crypto").lower()
        if market != "crypto":
            from providers import tradfi_data
            rows = tradfi_data.screener_rows(market)
            return jsonify({"rows": rows, "ts": int(time.time() * 1000)})
        exchange = (request.args.get("exchange") or "bybit").lower()
        top = min(int(request.args.get("top", 50)), 100)
        ex = get_exchange(exchange)

        tickers = _cached(("alltickers", exchange), 55, ex.fetch_tickers)
        perps = [(s, t) for s, t in tickers.items()
                 if s.endswith("/USDT:USDT") and t.get("quoteVolume")]
        perps.sort(key=lambda x: x[1]["quoteVolume"], reverse=True)
        perps = perps[:top]

        def fetch_frs():
            try:
                return ex.fetch_funding_rates([s for s, _ in perps])
            except Exception:
                return {}
        frs = _cached(("frs", exchange, top), 300, fetch_frs)

        rows = []
        for sym, t in perps:
            stats = _kline_stats(ex, sym)
            fr = frs.get(sym) or {}
            rows.append({
                "base": sym.split("/")[0],
                "symbol": sym,
                "last": _safe_float(t.get("last")),
                "pct24h": _safe_float(t.get("percentage")),
                "vol_usd": _safe_float(t.get("quoteVolume")),
                "funding": _safe_float(fr.get("fundingRate")),
                **{k: _safe_float(v) for k, v in stats.items()},
            })
        return jsonify({"rows": rows, "ts": int(time.time() * 1000)})
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── /des — visão geral do instrumento ────────────────────────────────────

@terminal_bp.get("/des")
def des():
    try:
        exchange = (request.args.get("exchange") or "bybit").lower()
        market = (request.args.get("market") or "crypto").lower()
        base = (request.args.get("symbol") or "").strip()
        if not base:
            return jsonify({"error": "symbol obrigatório"}), 400

        if market == "tradfi":
            from providers import tradfi_data
            return jsonify(tradfi_data.describe(base))

        try:
            ex = get_exchange(exchange)
            sym = normalize_symbol(base, exchange)
            mkt = ex.market(sym)
        except Exception:
            # modo auto: símbolo não existe na exchange -> tenta tradicional
            if market == "auto":
                from providers import tradfi_data
                return jsonify(tradfi_data.describe(base))
            raise
        ticker = ex.fetch_ticker(sym)

        try:
            oi = ex.fetch_open_interest(sym)
        except Exception:
            oi = {}
        try:
            fr = ex.fetch_funding_rate(sym)
        except Exception:
            fr = {}

        stats = _kline_stats(ex, sym)

        # histórico de funding 30d (cache parquet do módulo de custos)
        funding_hist = {"dates": [], "rates": []}
        try:
            from costs.funding import get_funding_events
            now = int(time.time() * 1000)
            evs = get_funding_events(exchange, sym, now - 30 * 86_400_000, now)
            funding_hist = {
                "dates": [int(e.timestamp) for e in evs],
                "rates": [float(e.rate) for e in evs],
            }
        except Exception:
            pass

        fees = {}
        try:
            from costs.config import DEFAULT_FEES
            f = DEFAULT_FEES.get(exchange)
            if f:
                fees = {"maker": float(f.maker), "taker": float(f.taker)}
        except Exception:
            pass

        limits = mkt.get("limits") or {}
        return jsonify({
            "kind": "crypto",
            "market": "crypto",
            "base": base.upper(),
            "symbol": sym,
            "exchange": exchange,
            "last": _safe_float(ticker.get("last")),
            "pct24h": _safe_float(ticker.get("percentage")),
            "high24": _safe_float(ticker.get("high")),
            "low24": _safe_float(ticker.get("low")),
            "vol_usd": _safe_float(ticker.get("quoteVolume")),
            "open_interest": _safe_float(oi.get("openInterestAmount")),
            "open_interest_usd": _safe_float(oi.get("openInterestValue")),
            "funding": _safe_float(fr.get("fundingRate")),
            "next_funding_ts": fr.get("fundingTimestamp") or fr.get("nextFundingTimestamp"),
            "funding_hist": funding_hist,
            "fees": fees,
            "min_qty": _safe_float(((limits.get("amount") or {}).get("min"))),
            "min_notional": _safe_float(((limits.get("cost") or {}).get("min"))),
            "contract_size": _safe_float(mkt.get("contractSize")),
            **{k: _safe_float(v) for k, v in stats.items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── Mercado macro/derivativos: juros, crédito, opções, book ─────────────

@terminal_bp.get("/rates")
def rates():
    try:
        from providers import markets_data
        out = markets_data.yield_curve()
        out = {**out, "credit": markets_data.credit_spreads()}
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/options")
def options():
    try:
        from providers import markets_data
        from engine import options_analytics
        base = (request.args.get("symbol") or "").strip()
        if not base:
            return jsonify({"error": "symbol obrigatório"}), 400
        expiry = request.args.get("expiry") or None
        chain = dict(markets_data.option_chain(base, expiry))
        return jsonify(options_analytics.add_greeks(chain))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/options/surface")
def options_surface():
    try:
        from engine import options_analytics
        base = (request.args.get("symbol") or "").strip()
        if not base:
            return jsonify({"error": "symbol obrigatório"}), 400
        return jsonify(options_analytics.vol_surface(base))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.post("/options/strategy")
def options_strategy():
    try:
        from engine import options_analytics
        payload = request.get_json(force=True) or {}
        return jsonify(options_analytics.strategy_eval(payload))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── GT — análise técnica multi-ativo ─────────────────────────────────────

@terminal_bp.post("/chart")
def chart():
    try:
        from providers import technical_data
        body = request.get_json(force=True) or {}
        symbols = body.get("symbols") or []
        if not symbols:
            return jsonify({"error": "symbols obrigatório"}), 400
        interval = body.get("interval") or "1d"
        bars = int(body.get("bars") or 500)
        if body.get("mode") == "compare":
            return jsonify(technical_data.compare_chart(
                symbols, interval, bars, int(body.get("corr_window") or 30)))
        return jsonify(technical_data.single_chart(
            symbols[0], interval, bars, body.get("studies") or []))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── ALTD — dados alternativos ────────────────────────────────────────────

@terminal_bp.get("/alt/indicators")
def alt_indicators():
    try:
        from providers import altdata
        return jsonify(altdata.indicators())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/supplychain")
def alt_supplychain():
    try:
        from providers import altdata
        return jsonify(altdata.supply_chain())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/traffic")
def alt_traffic():
    try:
        from providers import altdata
        return jsonify(altdata.traffic())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/climate")
def alt_climate():
    try:
        from providers import altdata
        return jsonify(altdata.climate())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/sectors")
def alt_sectors():
    try:
        from providers import altdata
        return jsonify(altdata.sector_metrics(request.args.get("s", "varejo")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/cryptomicro")
def alt_cryptomicro():
    try:
        from providers import altdata
        return jsonify(altdata.crypto_micro())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/onchain")
def alt_onchain():
    try:
        from providers import onchain_data
        return jsonify(onchain_data.overview())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/alt/onchain/coin")
def alt_onchain_coin():
    try:
        from providers import onchain_data
        return jsonify(onchain_data.coin(request.args.get("symbol", "")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── OMS/EMS — execução de ordens ─────────────────────────────────────────

@terminal_bp.get("/oms/accounts")
def oms_accounts():
    try:
        from engine import oms
        return jsonify(oms.accounts())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.post("/oms/pretrade")
def oms_pretrade():
    try:
        from engine import oms
        return jsonify(oms.pre_trade(request.get_json(force=True) or {}))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.post("/oms/orders")
def oms_submit():
    try:
        from engine import oms
        return jsonify(oms.submit_order(request.get_json(force=True) or {}))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.delete("/oms/orders/<order_id>")
def oms_cancel(order_id):
    try:
        from engine import oms
        return jsonify(oms.cancel_order(order_id))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/oms/blotter")
def oms_blotter():
    try:
        from engine import oms
        return jsonify(oms.blotter(request.args.get("account", "paper")))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/oms/tca")
def oms_tca():
    try:
        from engine import oms
        return jsonify(oms.tca(request.args.get("account", "paper")))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.post("/oms/reset")
def oms_reset():
    try:
        body = request.get_json(force=True) or {}
        if not body.get("confirm"):
            return jsonify({"error": "reset da conta paper exige confirm=true"}), 400
        from engine import oms
        return jsonify(oms.reset_paper())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── CDTY — painel de commodities ─────────────────────────────────────────

@terminal_bp.get("/cdty/overview")
def cdty_overview():
    try:
        from providers import commodities_data
        return jsonify(commodities_data.overview())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/cdty/curves")
def cdty_curves():
    try:
        from providers import commodities_data
        return jsonify({"curves": commodities_data.curves_meta()})
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/cdty/curve")
def cdty_curve():
    try:
        from providers import commodities_data
        root = (request.args.get("c") or "CL").strip()
        return jsonify(commodities_data.futures_curve(root))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/cdty/weather")
def cdty_weather():
    try:
        from providers import commodities_data
        return jsonify(commodities_data.weather())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/cdty/shipping")
def cdty_shipping():
    try:
        from providers import commodities_data
        return jsonify(commodities_data.shipping())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/cdty/inventories")
def cdty_inventories():
    try:
        from providers import commodities_data
        return jsonify(commodities_data.inventories())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/book")
def book():
    try:
        from providers import markets_data
        base = (request.args.get("symbol") or "").strip()
        if not base:
            return jsonify({"error": "symbol obrigatório"}), 400
        return jsonify(markets_data.order_book(
            base,
            (request.args.get("exchange") or "bybit").lower(),
            (request.args.get("market") or "crypto").lower(),
        ))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── EA — análise completa de empresa (estilo Bloomberg FA) ───────────────

@terminal_bp.get("/ea")
def ea():
    try:
        from engine import equity_analysis
        base = (request.args.get("symbol") or "").strip()
        if not base:
            return jsonify({"error": "symbol obrigatório"}), 400
        return jsonify(equity_analysis.analyze(base))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── EQS — screening fundamentalista (Yahoo screener server-side) ────────

@terminal_bp.get("/eqs/meta")
def eqs_meta():
    try:
        from providers import eqs_data
        return jsonify(eqs_data.meta())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.post("/eqs/equity")
def eqs_equity():
    try:
        from providers import eqs_data
        filters = request.get_json(force=True) or {}
        return jsonify(eqs_data.run_equity_screen(filters))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


@terminal_bp.get("/eqs/funds")
def eqs_funds():
    try:
        from providers import eqs_data
        screen = request.args.get("screen") or "top_etfs_us"
        return jsonify(eqs_data.run_fund_screen(screen))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500


# ── Alertas (preço/funding) ──────────────────────────────────────────────

_ALERTS_FILE = Path(__file__).parents[1] / "data" / "alerts_data.json"
_alerts_lock = threading.Lock()
_ALERT_KINDS = ("price_above", "price_below", "funding_above", "funding_below",
                "signal_score_above", "signal_score_below")


def _alerts_load() -> list:
    if not _ALERTS_FILE.exists():
        return []
    try:
        return json.loads(_ALERTS_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []


def _alerts_save(alerts: list) -> None:
    _ALERTS_FILE.write_text(json.dumps(alerts, ensure_ascii=False, indent=1),
                            encoding="utf-8")


@terminal_bp.get("/alerts")
def alerts_list():
    _ensure_watcher()
    with _alerts_lock:
        return jsonify({"alerts": _alerts_load()})


@terminal_bp.post("/alerts")
def alerts_create():
    body = request.get_json(force=True) or {}
    kind = body.get("kind")
    symbol = (body.get("symbol") or "").strip().upper()
    level = _safe_float(body.get("level"))
    market = (body.get("market") or "crypto").lower()
    if kind not in _ALERT_KINDS:
        return jsonify({"error": f"kind deve ser um de {_ALERT_KINDS}"}), 400
    if not symbol or level is None:
        return jsonify({"error": "symbol e level obrigatórios"}), 400
    if market == "tradfi" and kind.startswith("funding"):
        return jsonify({"error": "funding não existe no mercado tradicional"}), 400
    alert = {
        "id": uuid.uuid4().hex[:10],
        "symbol": symbol,
        "market": market,
        "exchange": (body.get("exchange") or "bybit").lower(),
        "kind": kind,
        "level": level,
        "note": (body.get("note") or "")[:200],
        "active": True,
        "created_at": int(time.time() * 1000),
        "triggered_at": None,
        "trigger_value": None,
    }
    with _alerts_lock:
        alerts = _alerts_load()
        alerts.append(alert)
        _alerts_save(alerts)
    _ensure_watcher()
    return jsonify({"alert": alert})


@terminal_bp.delete("/alerts/<alert_id>")
def alerts_delete(alert_id):
    with _alerts_lock:
        alerts = _alerts_load()
        alerts = [a for a in alerts if a["id"] != alert_id]
        _alerts_save(alerts)
    return jsonify({"ok": True})


_watcher_started = False
_watcher_lock = threading.Lock()


def _ensure_watcher():
    """Inicia a thread de checagem sob demanda (evita duplicar no reloader:
    só requests reais chegam aqui)."""
    global _watcher_started
    with _watcher_lock:
        if _watcher_started:
            return
        threading.Thread(target=_alert_watcher, daemon=True,
                         name="terminal-alert-watcher").start()
        threading.Thread(target=_signal_alert_watcher, daemon=True,
                         name="terminal-signal-alert-watcher").start()
        _watcher_started = True


def _alert_watcher():
    while True:
        time.sleep(30)
        try:
            with _alerts_lock:
                alerts = _alerts_load()
            active = [a for a in alerts if a.get("active")
                      and not a["kind"].startswith("signal_score")]
            if not active:
                continue
            # agrupa por exchange; 1 fetch de tickers+funding por exchange
            by_ex: dict = {}
            tradfi_syms: set = set()
            for a in active:
                if a.get("market") == "tradfi":
                    tradfi_syms.add(a["symbol"])
                else:
                    by_ex.setdefault(a["exchange"], set()).add(a["symbol"])
            quotes: dict = {}
            if tradfi_syms:
                try:
                    from providers import tradfi_data
                    tq = tradfi_data.quotes(list(tradfi_syms))
                    for s, row in tq.items():
                        quotes[("tradfi", s)] = {"price": row.get("last"),
                                                 "funding": None}
                except Exception:
                    pass
            for exch, bases in by_ex.items():
                ex = get_exchange(exch)
                syms = {b: normalize_symbol(b, exch) for b in bases}
                tickers = ex.fetch_tickers(list(syms.values()))
                try:
                    frs = ex.fetch_funding_rates(list(syms.values()))
                except Exception:
                    frs = {}
                for b, s in syms.items():
                    quotes[(exch, b)] = {
                        "price": _safe_float((tickers.get(s) or {}).get("last")),
                        "funding": _safe_float((frs.get(s) or {}).get("fundingRate")),
                    }
            changed = False
            for a in active:
                qkey = ("tradfi", a["symbol"]) if a.get("market") == "tradfi" \
                    else (a["exchange"], a["symbol"])
                q = quotes.get(qkey) or {}
                v = q.get("funding") if a["kind"].startswith("funding") else q.get("price")
                if v is None:
                    continue
                hit = (v >= a["level"] if a["kind"].endswith("above") else v <= a["level"])
                if hit:
                    a["active"] = False
                    a["triggered_at"] = int(time.time() * 1000)
                    a["trigger_value"] = v
                    changed = True
            if changed:
                with _alerts_lock:
                    # regrava preservando alertas criados durante o ciclo
                    cur = _alerts_load()
                    by_id = {a["id"]: a for a in alerts}
                    out = [by_id.get(c["id"], c) for c in cur]
                    _alerts_save(out)
        except Exception:
            pass  # ciclo seguinte tenta de novo


def _signal_alert_watcher():
    """Slow multifactor checks are isolated so price/funding alerts stay fast."""
    while True:
        time.sleep(60)
        try:
            from providers import intelligence_data
            # mantém o ranking quente (rebuild a cada 15 min) e vira "vigia":
            # transições de sinal e divergências novas viram alertas disparados
            intelligence_data.ranking()
            events = intelligence_data.pop_events()
            if events:
                now = int(time.time() * 1000)
                rows = [{"id": uuid.uuid4().hex[:10], "symbol": e["symbol"],
                         "market": "signal", "exchange": None, "kind": e["kind"],
                         "level": None, "note": e["detail"], "active": False,
                         "created_at": now, "triggered_at": now,
                         "trigger_value": None} for e in events]
                with _alerts_lock:
                    cur = _alerts_load() + rows
                    # ponytail: só os 200 eventos de sinal mais recentes ficam
                    auto = [a for a in cur if a["kind"] in ("signal_change", "divergence_new")]
                    drop = {a["id"] for a in sorted(auto, key=lambda a: a["triggered_at"])[:-200]}
                    _alerts_save([a for a in cur if a["id"] not in drop])

            with _alerts_lock:
                alerts = _alerts_load()
            active = [a for a in alerts if a.get("active")
                      and a["kind"].startswith("signal_score")]
            if not active:
                continue
            scores = {}
            for symbol in {a["symbol"] for a in active}:
                try:
                    scores[symbol] = intelligence_data.analyze(symbol)["signal"]["score"]
                except Exception:
                    pass
            changed = False
            for a in active:
                value = scores.get(a["symbol"])
                if value is None:
                    continue
                hit = value >= a["level"] if a["kind"].endswith("above") else value <= a["level"]
                if hit:
                    a.update(active=False, triggered_at=int(time.time()*1000),
                             trigger_value=value)
                    changed = True
            if changed:
                with _alerts_lock:
                    current = _alerts_load()
                    updates = {a["id"]: a for a in alerts}
                    _alerts_save([updates.get(a["id"], a) for a in current])
        except Exception:
            pass


# ── /news — RSS agregado ─────────────────────────────────────────────────

_NEWS_SOURCES = [
    # (nome, url, categoria)
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "crypto"),
    ("Cointelegraph", "https://cointelegraph.com/rss", "crypto"),
    ("Decrypt", "https://decrypt.co/feed", "crypto"),
    ("Livecoins", "https://livecoins.com.br/feed/", "crypto"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex", "markets"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "markets"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html", "markets"),
    ("InfoMoney", "https://www.infomoney.com.br/feed/", "markets"),
    ("OilPrice", "https://oilprice.com/rss/main", "commodities"),
    ("Mining.com", "https://www.mining.com/feed/", "commodities"),
]


def _parse_feed(name, url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (terminal-graph)"})
    with urlopen(req, timeout=8) as resp:
        root = ET.fromstring(resp.read())
    items = []
    # RSS 2.0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        ts = None
        if pub:
            try:
                ts = int(parsedate_to_datetime(pub).timestamp() * 1000)
            except (TypeError, ValueError):
                pass
        if title and link:
            items.append({"title": title, "link": link, "source": name, "ts": ts})
    # Atom fallback
    if not items:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", namespaces=ns) or "").strip()
            link_el = entry.find("a:link", ns)
            link = link_el.get("href") if link_el is not None else ""
            when = entry.findtext("a:updated", namespaces=ns)
            ts = None
            if when:
                try:
                    from datetime import datetime
                    ts = int(datetime.fromisoformat(when.replace("Z", "+00:00"))
                             .timestamp() * 1000)
                except ValueError:
                    pass
            if title and link:
                items.append({"title": title, "link": link, "source": name, "ts": ts})
    return items


@terminal_bp.get("/news")
def news():
    cat = (request.args.get("cat") or "all").lower()

    def fetch():
        items, failed = [], []
        for name, url, source_cat in _NEWS_SOURCES:
            try:
                got = _parse_feed(name, url)[:25]
                for it in got:
                    it["cat"] = source_cat
                items.extend(got)
            except Exception:
                failed.append(name)
        items.sort(key=lambda x: x["ts"] or 0, reverse=True)
        return {"items": items, "failed_sources": failed}

    try:
        data = _cached(("news",), 600, fetch)
        items = data["items"]
        if cat in ("crypto", "markets", "commodities"):
            items = [it for it in items if it.get("cat") == cat]
        q = (request.args.get("q") or "").strip().lower()
        if q:
            terms = [w for w in q.split("|") if w]
            items = [it for it in items
                     if any(w in it["title"].lower() for w in terms)]
        return jsonify({"items": items[:100], "failed_sources": data["failed_sources"]})
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500

@terminal_bp.get("/seasonality")
def seasonality():
    try:
        from providers import seasonality_data
        return jsonify(seasonality_data.analyze(request.args.get("symbol", "")))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 400

@terminal_bp.get("/intelligence")
def intelligence():
    try:
        from providers import intelligence_data
        return jsonify(intelligence_data.analyze(request.args.get("symbol", "BTC")))
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 400

@terminal_bp.get("/intelligence/ranking")
def intelligence_ranking():
    try:
        from providers import intelligence_data
        return jsonify(intelligence_data.ranking())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 400

@terminal_bp.get("/intelligence/tracking")
def intelligence_tracking():
    try:
        from providers import intelligence_data
        return jsonify(intelligence_data.tracking())
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 400

@terminal_bp.get("/calendar")
def market_calendar():
    try:
        from providers import calendar_data
        symbols=[x for x in request.args.get("symbols","").upper().split(",") if x]
        return jsonify(calendar_data.events(symbols))
    except Exception as e:return jsonify({"error":str(e)[:300]}),400

@terminal_bp.get("/liquidity")
def liquidity():
    try:
        from providers import liquidity_data
        return jsonify(liquidity_data.snapshot())
    except Exception as e:return jsonify({"error":str(e)[:300]}),400

@terminal_bp.post("/portfolio-lab")
def portfolio_lab_route():
    try:
        from engine import portfolio_lab
        body=request.get_json(force=True) or {}
        return jsonify(portfolio_lab.analyze(body.get("symbols") or [],body.get("years",3)))
    except Exception as e:return jsonify({"error":str(e)[:300]}),400
