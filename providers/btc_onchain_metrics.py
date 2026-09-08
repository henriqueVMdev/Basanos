"""Bitcoin market/on-chain metrics with honest source attribution.

Public metrics are calculated locally. Entity-labelled metrics are requested from
Glassnode when ``GLASSNODE_API_KEY`` is configured.  CryptoQuant-only series can
be connected without code changes through the ``CRYPTOQUANT_*_URL`` variables.
"""
from __future__ import annotations

import os
import json
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_UA = {"User-Agent": "GraphQuantLab/1.0"}
_PAYLOAD_CACHE = None
_PAYLOAD_LOCK = threading.Lock()


def _series(rows, value_key="v"):
    points = [(int(r["t"]) * 1000, r.get(value_key)) for r in rows
              if r.get("t") is not None and isinstance(r.get(value_key), (int, float))]
    return {"ts": [p[0] for p in points], "values": [p[1] for p in points]}


def _glassnode(path: str, days=730):
    key = os.getenv("GLASSNODE_API_KEY", "").strip()
    if not key:
        return None
    r = requests.get("https://api.glassnode.com/v1/metrics/" + path,
                     params={"a": "BTC", "i": "24h", "s": int(time.time()) - days * 86400,
                             "api_key": key}, headers=_UA, timeout=30)
    r.raise_for_status()
    return _series(r.json())


# bitcoin-data.com: métricas on-chain BTC gratuitas, sem chave (fallback do Glassnode)
# Limite: 10 req/hora → cache em disco de 12h (métricas diárias) e stale em falha.
_BD_METRICS = {
    "nupl": ("nupl", "nupl"),
    "sth_realized_price": ("sth-realized-price", "sthRealizedPrice"),
    "lth_realized_price": ("lth-realized-price", "lthRealizedPrice"),
    "sth_mvrv": ("sth-mvrv", "sthMvrv"),
    "sth_sopr": ("sth-sopr", "sthSopr"),
}
_BD_LOCK = threading.Lock()
_BD_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "bitcoin_data_cache.json")


def _bd_cache_load():
    try:
        with open(_BD_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _bitcoin_data(endpoint: str, value_key: str, days=730, ttl_s=12 * 3600):
    from datetime import date, timedelta
    with _BD_LOCK:
        cache = _bd_cache_load()
        hit = cache.get(endpoint)
        if hit and time.time() - hit["fetched_at"] < ttl_s:
            return hit["series"]
        try:
            params = {"startday": (date.today() - timedelta(days=days)).isoformat(),
                      "endday": date.today().isoformat()}
            r = requests.get(f"https://bitcoin-data.com/v1/{endpoint}", params=params,
                             headers={**_UA, "Accept": "application/json"}, timeout=30)
            r.raise_for_status()
            pts = [(int(row["unixTs"]) * 1000, float(row[value_key])) for row in r.json()
                   if row.get(value_key) is not None]
            series = {"ts": [p[0] for p in pts], "values": [p[1] for p in pts]}
            cache[endpoint] = {"fetched_at": time.time(), "series": series}
            with open(_BD_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            time.sleep(1)
            return series
        except Exception:
            if hit:
                return hit["series"]  # dado de ontem > gráfico vazio
            raise


def _glassnode_or_free(key: str, path: str):
    s = _glassnode(path)
    if s:
        return {**s, "source": "Glassnode"}
    endpoint, value_key = _BD_METRICS[key]
    # 10 anos: 1 requisição igual, e o backtest causal precisa de história longa
    s = _bitcoin_data(endpoint, value_key, days=3650)
    return {**s, "source": "bitcoin-data.com"} if s and s["ts"] else None


def _configured_cq(metric_id: str):
    """Read a CryptoQuant-compatible JSON series from a configured URL.

    Accepted records: {timestamp|t|date, value|v}. This deliberately avoids
    hard-coding private/plan-dependent CryptoQuant routes.
    """
    url = os.getenv(f"CRYPTOQUANT_{metric_id.upper()}_URL", "").strip()
    if not url:
        return None
    key = os.getenv("CRYPTOQUANT_API_KEY", "").strip()
    headers = {**_UA, **({"Authorization": f"Bearer {key}"} if key else {})}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    body = r.json()
    rows = body.get("result", {}).get("data") if isinstance(body, dict) else body
    rows = rows or (body.get("data") if isinstance(body, dict) else []) or []
    pts = []
    for row in rows:
        t = row.get("timestamp", row.get("t", row.get("date")))
        v = row.get("value", row.get("v"))
        try:
            if isinstance(t, str):
                from datetime import datetime
                t = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
            t, v = float(t), float(v)
            if t > 1e12:
                t /= 1000
            pts.append((int(t) * 1000, v))
        except (TypeError, ValueError):
            continue
    pts.sort()
    return {"ts": [x[0] for x in pts], "values": [x[1] for x in pts]}


def _pi_cycle():
    # Binance public daily candles; fetch backwards because each page is capped at 1,000.
    rows, end = [], None
    for _ in range(3):
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
        if end is not None:
            params["endTime"] = end
        r = requests.get("https://api.binance.com/api/v3/klines", params=params,
                         headers=_UA, timeout=30)
        r.raise_for_status()
        page = [(int(x[0]), float(x[4])) for x in r.json()]
        if not page:
            break
        rows = page + rows
        end = page[0][0] - 1
    rows = sorted(dict(rows).items())
    vals = [x[1] for x in rows]
    def ma(i, n):
        return sum(vals[i - n + 1:i + 1]) / n if i + 1 >= n else None
    start = max(0, len(rows) - 2920)
    return {"ts": [x[0] for x in rows[start:]], "price": vals[start:],
            "dma111": [ma(i, 111) for i in range(start, len(rows))],
            "dma350x2": [(ma(i, 350) * 2 if ma(i, 350) is not None else None)
                         for i in range(start, len(rows))],
            "source": "Binance BTCUSDT; cálculo local (111DMA e 2×350DMA)"}


def _open_interest():
    import ccxt
    exchanges = {"Binance": ccxt.binanceusdm, "Bybit": ccxt.bybit, "OKX": ccxt.okx}
    rows, errors, contracts, excluded = [], [], [], []
    for name, cls in exchanges.items():
        try:
            ex = cls({"enableRateLimit": True})
            markets = ex.load_markets()
            symbols = [s for s, m in markets.items() if m.get("active") is not False
                       and m.get("base") == "BTC" and (m.get("swap") or m.get("future"))]
            tickers = {}
            for market_type in ("swap", "future"):
                group = [s for s in symbols if markets[s].get(market_type)]
                if not group:
                    continue
                try:
                    tickers.update(ex.fetch_tickers(group))
                except Exception:
                    # A missing bulk ticker only excludes affected fallbacks;
                    # it never invents a USD conversion.
                    pass
            total = 0.0
            seen_ids = set()
            for symbol in symbols:
                try:
                    market = markets[symbol]
                    market_id = str(market.get("id") or symbol)
                    if market_id in seen_ids:
                        excluded.append({"exchange": name, "symbol": symbol,
                                         "reason": "market id duplicado"})
                        continue
                    seen_ids.add(market_id)
                    oi = ex.fetch_open_interest(symbol)
                    usd = oi.get("openInterestValue")
                    if usd is None:
                        amount = oi.get("openInterestAmount")
                        ticker = tickers.get(symbol) or {}
                        price = ((oi.get("info") or {}).get("markPrice") or oi.get("markPrice")
                                 or ticker.get("last") or ticker.get("close"))
                        if amount and not price and market.get("linear"):
                            try:
                                one_ticker = ex.fetch_ticker(symbol)
                                price = one_ticker.get("last") or one_ticker.get("close")
                            except Exception:
                                pass
                        # Unified amount is normally base-denominated for linear
                        # contracts. Inverse fallbacks are excluded unless the
                        # exchange provides openInterestValue in quote currency.
                        usd = (float(amount) * float(price)
                               if amount and price and market.get("linear") else None)
                    usd = float(usd) if usd is not None else None
                    if usd is None or not math.isfinite(usd) or usd <= 0:
                        excluded.append({"exchange": name, "symbol": symbol,
                                         "reason": "sem valor de OI normalizado em USD"})
                        continue
                    total += usd
                    contracts.append({"exchange": name, "symbol": symbol,
                                      "market_id": market_id, "oi_usd": usd,
                                      "linear": bool(market.get("linear")),
                                      "inverse": bool(market.get("inverse")),
                                      "future": bool(market.get("future")),
                                      "swap": bool(market.get("swap")),
                                      "quote": market.get("quote"),
                                      "settle": market.get("settle")})
                except Exception:
                    continue
            rows.append({"exchange": name, "oi_usd": total or None, "symbols": len(symbols)})
        except Exception as exc:
            errors.append(f"{name}: {str(exc)[:80]}")
    return {"total_usd": sum(x["oi_usd"] or 0 for x in rows) or None,
            "exchanges": rows, "errors": errors,
            "contracts": contracts, "excluded": excluded,
            "scope": "BTC futures/perpetuals disponíveis em Binance, Bybit e OKX",
            "source": "APIs públicas das exchanges via CCXT"}


def _sth_sopr_mvrv_indicator(mvrv: dict, sopr: dict) -> dict:
    """Reproduce Checkonchain's break-even oscillator transformation."""
    mv = dict(zip(mvrv["ts"], mvrv["values"]))
    sp = dict(zip(sopr["ts"], sopr["values"]))
    ts = sorted(mv.keys() & sp.keys())
    m = [mv[t] for t in ts]
    # The chart doubles SOPR's distance from 1 so both signals share one axis.
    s2 = [1.0 + 2.0 * (sp[t] - 1.0) for t in ts]
    return {
        "ts": ts,
        "mvrv_positive": [max(1.0, v) for v in m],
        "mvrv_negative": [min(1.0, v) for v in m],
        "sopr2_positive": [max(1.0, v) for v in s2],
        "sopr2_negative": [min(1.0, v) for v in s2],
        "break_even": 1.0,
        "latest": {"sth_mvrv": m[-1] if m else None,
                   "sth_sopr": sp[ts[-1]] if ts else None,
                   "combined": ((m[-1] + s2[-1]) / 2 if m else None)},
        "formula": "MVRV-STH; 2x STH-SOPR = 1 + 2 × (STH-SOPR − 1)",
        "source": "Glassnode; transformação reproduzida do gráfico Checkonchain",
    }


def payload(ttl=900):
    """Cached aggregate; the lock prevents duplicate cold refreshes."""
    global _PAYLOAD_CACHE
    now = time.time()
    if _PAYLOAD_CACHE and now - _PAYLOAD_CACHE[0] < ttl:
        return _PAYLOAD_CACHE[1]
    with _PAYLOAD_LOCK:
        now = time.time()
        if _PAYLOAD_CACHE and now - _PAYLOAD_CACHE[0] < ttl:
            return _PAYLOAD_CACHE[1]
        value = _build_payload()
        _PAYLOAD_CACHE = (time.time(), value)
        return value


def clear_cache():
    global _PAYLOAD_CACHE
    with _PAYLOAD_LOCK:
        _PAYLOAD_CACHE = None


def _build_payload():
    specs = {
        "nupl": ("indicators/net_unrealized_profit_loss", "Glassnode"),
        "sth_realized_price": ("indicators/realized_price_less_155", "Glassnode"),
        "lth_realized_price": ("indicators/realized_price_more_155", "Glassnode"),
        "sth_mvrv": ("market/mvrv_less_155", "Glassnode"),
        "sth_sopr": ("indicators/sopr_less_155", "Glassnode"),
    }
    cq = {
        "exchange_whale_ratio": "Exchange Whale Ratio",
        "estimated_leverage_ratio": "Estimated Leverage Ratio — All Exchanges",
        "retail_demand_30d": "Retail Investor Demand 30D Change ($0–$10K)",
        "whale_realized_price": "Realized Price: Short-term vs Long-term Whales",
        "whale_last_active": "Whale Last Active",
    }
    out = {"series": {}, "open_interest": None, "pi_cycle": None, "unavailable": [],
           "ts": int(time.time() * 1000)}
    jobs = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs[pool.submit(_pi_cycle)] = ("pi_cycle", "public")
        jobs[pool.submit(_open_interest)] = ("open_interest", "public")
        for key, (path, source) in specs.items():
            jobs[pool.submit(_glassnode_or_free, key, path)] = (key, source)
        for key in cq:
            jobs[pool.submit(_configured_cq, key)] = (key, "CryptoQuant")
        for future in as_completed(jobs):
            key, source = jobs[future]
            try:
                value = future.result()
                if key in ("pi_cycle", "open_interest"):
                    out[key] = value
                elif value:
                    out["series"][key] = {"source": source, **value}
                else:
                    label = cq.get(key, key.replace("_", " ").title())
                    out["unavailable"].append({"id": key, "label": label,
                                               "reason": f"configure a credencial/URL de {source}"})
            except Exception as exc:
                out["unavailable"].append({"id": key, "label": cq.get(key, key),
                                           "reason": str(exc)[:140]})
    mvrv, sopr = out["series"].get("sth_mvrv"), out["series"].get("sth_sopr")
    if mvrv and sopr:
        out["sth_sopr_mvrv_indicator"] = _sth_sopr_mvrv_indicator(mvrv, sopr)
    else:
        out["sth_sopr_mvrv_indicator"] = None
    return out


if __name__ == "__main__":
    for key, (endpoint, value_key) in _BD_METRICS.items():
        s = _bitcoin_data(endpoint, value_key, days=30)
        assert len(s["ts"]) > 10 and s["ts"] == sorted(s["ts"]), (key, s)
        print(f"{key}: {len(s['ts'])} pontos, último = {s['values'][-1]}")
