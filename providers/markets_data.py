"""
Dados de mercado macro/derivativos via Yahoo (yfinance) e ccxt:
curva de juros US, spreads de crédito, volatilidade implícita/histórica,
chain de opções e book de ofertas (L2 cripto; bid/ask top-of-book p/ tradfi).

Spreads de crédito: tenta FRED (CSV público); se a rede bloquear, cai para
proxy honesto = yield de distribuição de ETFs (HYG/LQD) menos treasury de
duration equivalente — o payload marca "source" para a UI rotular.
"""

from __future__ import annotations

import math
import time

import numpy as np

from common import ttl_cache, to_float

_cached = ttl_cache()


# curva de juros US
_CURVE = [
    ("3M", "^IRX", 0.25), ("2A", "2YY=F", 2), ("5A", "^FVX", 5),
    ("10A", "^TNX", 10), ("30A", "^TYX", 30),
]


def yield_curve() -> dict:
    def fetch():
        import yfinance as yf
        points = []
        for label, sym, years in _CURVE:
            try:
                h = yf.Ticker(sym).history(period="1y", interval="1d")["Close"].dropna()
                if h.empty:
                    continue
                points.append({
                    "label": label, "symbol": sym, "years": years,
                    "now": to_float(h.iloc[-1]),
                    "m1": to_float(h.iloc[-22]) if len(h) > 22 else None,
                    "y1": to_float(h.iloc[0]) if len(h) > 200 else None,
                })
            except Exception:
                continue
        vix = None
        try:
            import yfinance as yf
            vix = to_float(yf.Ticker("^VIX").fast_info["last_price"])
        except Exception:
            pass
        out = {"points": points, "vix": vix, "ts": int(time.time() * 1000)}
        if len(points) >= 2:
            by = {p["label"]: p["now"] for p in points}
            if by.get("10A") and by.get("2A"):
                out["spread_10y2y"] = round(by["10A"] - by["2A"], 3)
            if by.get("10A") and by.get("3M"):
                out["spread_10y3m"] = round(by["10A"] - by["3M"], 3)
        return out

    return _cached(("curve",), 900, fetch)


# spreads de crédito
def _fred_series(series_id, start):
    import requests
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}&cosd={start}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    dates, vals = [], []
    for line in r.text.strip().split("\n")[1:]:
        d, v = line.split(",")[:2]
        fv = to_float(v)
        if fv is not None:
            dates.append(d)
            vals.append(fv)
    return dates, vals


def credit_spreads() -> dict:
    def fetch():
        from datetime import date, timedelta
        start = (date.today() - timedelta(days=365)).isoformat()
        # OAS oficial (ICE BofA via FRED)
        try:
            hy_d, hy_v = _fred_series("BAMLH0A0HYM2", start)
            ig_d, ig_v = _fred_series("BAMLC0A0CM", start)
            return {
                "source": "fred",
                "hy": {"dates": hy_d, "values": hy_v, "now": hy_v[-1]},
                "ig": {"dates": ig_d, "values": ig_v, "now": ig_v[-1]},
                "ts": int(time.time() * 1000),
            }
        except Exception:
            pass
        # Fallback: yield de distribuicao do ETF - treasury de duration equivalente
        import yfinance as yf
        curve = {p["label"]: p["now"] for p in yield_curve()["points"]}
        out = {"source": "proxy_etf", "ts": int(time.time() * 1000)}
        for key, etf, tsy_label in (("hy", "HYG", "5A"), ("ig", "LQD", "10A")):
            try:
                info = yf.Ticker(etf).info or {}
                y = to_float(info.get("yield") or info.get("trailingAnnualDividendYield"))
                y = y * 100 if y is not None and y < 1 else y
                tsy = curve.get(tsy_label)
                out[key] = {
                    "etf": etf, "etf_yield": y, "treasury": tsy,
                    "now": round(y - tsy, 2) if y is not None and tsy else None,
                }
            except Exception:
                out[key] = None
        return out

    return _cached(("credit",), 1800, fetch)


# volatilidade + opções
def _hist_vol(t, window=30):
    """Vol realizada anualizada (%) das ultimas `window` barras diarias."""
    h = t.history(period="6mo", interval="1d")["Close"].dropna()
    if len(h) < window + 1:
        return None
    rets = np.log(h / h.shift()).dropna().tail(window)
    return float(rets.std(ddof=1) * math.sqrt(252) * 100)


def _chain_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "strike": to_float(r.get("strike")),
            "last": to_float(r.get("lastPrice")),
            "bid": to_float(r.get("bid")),
            "ask": to_float(r.get("ask")),
            "volume": to_float(r.get("volume")),
            "oi": to_float(r.get("openInterest")),
            "iv": to_float(r.get("impliedVolatility")),
            "itm": bool(r.get("inTheMoney")),
        })
    return rows


def option_chain(symbol: str, expiry: str | None = None) -> dict:
    from providers import tradfi_data
    yf_sym = tradfi_data.resolve(symbol)

    def fetch():
        import yfinance as yf
        t = yf.Ticker(yf_sym)
        expiries = list(t.options or [])
        if not expiries:
            return {"error": f"{yf_sym} não tem opções listadas no Yahoo"}
        exp = expiry if expiry in expiries else expiries[0]
        ch = t.option_chain(exp)
        spot = to_float(t.fast_info["last_price"])

        calls = _chain_rows(ch.calls)
        puts = _chain_rows(ch.puts)

        # IV ATM = média call+put no strike mais próximo do spot
        atm_iv = None
        if spot:
            ivs = []
            for side in (calls, puts):
                cands = [r for r in side if r["strike"] and r["iv"]]
                if cands:
                    ivs.append(min(cands, key=lambda r: abs(r["strike"] - spot))["iv"])
            if ivs:
                atm_iv = sum(ivs) / len(ivs) * 100

        from datetime import date
        dte = (date.fromisoformat(exp) - date.today()).days
        hv = _hist_vol(t)

        return {
            "symbol": symbol.upper(), "yf_symbol": yf_sym,
            "spot": spot, "expiries": expiries, "expiry": exp, "dte": dte,
            "calls": calls, "puts": puts,
            "atm_iv": round(atm_iv, 2) if atm_iv else None,
            "hist_vol30": round(hv, 2) if hv else None,
            "ts": int(time.time() * 1000),
        }

    return _cached(("chain", yf_sym, expiry), 600, fetch)


# book de ofertas + negócios recentes
def order_book(symbol: str, exchange: str = "bybit", market: str = "crypto") -> dict:
    if market == "tradfi":
        # ações/ETFs: só top-of-book (bid/ask) está disponível de graça
        from providers import tradfi_data
        yf_sym = tradfi_data.resolve(symbol)

        def fetch():
            import yfinance as yf
            info = yf.Ticker(yf_sym).info or {}
            return {
                "market": "tradfi", "symbol": symbol.upper(), "yf_symbol": yf_sym,
                "bid": to_float(info.get("bid")), "ask": to_float(info.get("ask")),
                "bid_size": to_float(info.get("bidSize")), "ask_size": to_float(info.get("askSize")),
                "last": to_float(info.get("currentPrice") or info.get("regularMarketPrice")),
                "volume": to_float(info.get("volume") or info.get("regularMarketVolume")),
                "note": "book completo (L2) não é público p/ ações; bid/ask top-of-book",
                "ts": int(time.time() * 1000),
            }

        return _cached(("tob", yf_sym), 15, fetch)

    from providers.market_data import get_exchange, normalize_symbol
    ex = get_exchange(exchange)
    sym = normalize_symbol(symbol, exchange)
    ob = ex.fetch_order_book(sym, limit=15)
    trades = ex.fetch_trades(sym, limit=40)
    return {
        "market": "crypto", "symbol": symbol.upper(), "pair": sym,
        "bids": [[to_float(p), to_float(q)] for p, q, *_ in ob.get("bids", [])[:15]],
        "asks": [[to_float(p), to_float(q)] for p, q, *_ in ob.get("asks", [])[:15]],
        "trades": [{
            "ts": t.get("timestamp"), "side": t.get("side"),
            "price": to_float(t.get("price")), "qty": to_float(t.get("amount")),
        } for t in reversed(trades)],
        "ts": int(time.time() * 1000),
    }
