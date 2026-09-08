"""
Dados de mercado tradicional (ações US, commodities, forex, índices) via
yfinance — mesmo provedor já usado como fallback do backtest, sem chave de API.

Usado pelo terminal_api para estender /watch, /spark, /screener, /des e os
alertas ao mercado tradicional. Cache TTL próprio (dados Yahoo são atrasados
~15min; polling agressivo não ganha nada).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

_CACHE: dict = {}


def _cached(key, ttl_s, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl_s:
        return hit[1]
    data = fn()
    _CACHE[key] = (time.time(), data)
    return data


def _f(v):
    try:
        f = float(v)
        return f if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None


# ── universos e aliases ──────────────────────────────────────────────────

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "AVGO",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "NFLX", "JNJ",
    "ABBV", "CRM", "BAC", "ORCL", "MRK", "CVX", "KO", "AMD", "PEP", "TMO",
    "WMT", "ADBE", "CSCO", "ACN", "MCD", "LIN", "ABT", "PM", "IBM", "GE",
    "TXN", "INTU", "QCOM", "DHR", "VZ", "AMGN", "CAT", "PFE", "NEE", "SPGI",
    "RTX", "HON", "UNP", "AMAT", "LOW", "T", "BLK", "COP", "SYK", "BSX",
    "SCHW", "ETN", "PLD", "BMY", "DE", "MDT", "ADP", "GILD", "LMT", "ADI",
    "SBUX", "MMC", "BA", "CB", "MU", "INTC", "UPS", "SO", "ELV", "PGR",
    "MO", "NKE", "ISRG", "DUK", "CI", "VRTX", "ZTS", "SHW", "CL", "EQIX",
    "ITW", "PANW", "CME", "WM", "FDX", "EOG", "TGT", "CSX", "GD", "HCA",
]

COMMODITIES = [
    ("GC=F", "Ouro"), ("SI=F", "Prata"), ("PL=F", "Platina"), ("HG=F", "Cobre"),
    ("CL=F", "Petróleo WTI"), ("BZ=F", "Brent"), ("NG=F", "Gás Natural"),
    ("ZC=F", "Milho"), ("ZW=F", "Trigo"), ("ZS=F", "Soja"),
    ("KC=F", "Café"), ("SB=F", "Açúcar"), ("CC=F", "Cacau"),
    ("CT=F", "Algodão"), ("LE=F", "Boi Gordo"),
]

FOREX = [
    ("EURUSD=X", "EUR/USD"), ("GBPUSD=X", "GBP/USD"), ("USDJPY=X", "USD/JPY"),
    ("USDCHF=X", "USD/CHF"), ("AUDUSD=X", "AUD/USD"), ("USDCAD=X", "USD/CAD"),
    ("NZDUSD=X", "NZD/USD"), ("BRL=X", "USD/BRL"), ("EURBRL=X", "EUR/BRL"),
    ("MXN=X", "USD/MXN"), ("CNY=X", "USD/CNY"), ("EURGBP=X", "EUR/GBP"),
    ("EURJPY=X", "EUR/JPY"), ("DX-Y.NYB", "Índice Dólar (DXY)"),
]

INDICES = [
    ("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "Nasdaq Composite"),
    ("^NDX", "Nasdaq 100"), ("^RUT", "Russell 2000"), ("^VIX", "VIX"),
    ("^BVSP", "Ibovespa"), ("^FTSE", "FTSE 100"), ("^GDAXI", "DAX"),
    ("^FCHI", "CAC 40"), ("^N225", "Nikkei 225"), ("^HSI", "Hang Seng"),
    ("^TNX", "US 10Y (yield)"),
]

UNIVERSES = {
    "stocks": [(s, s) for s in STOCKS],
    "commodities": COMMODITIES,
    "forex": FOREX,
    "indices": INDICES,
}

# nomes amigáveis digitáveis no command line / formulários
ALIASES = {
    "OURO": "GC=F", "GOLD": "GC=F", "PRATA": "SI=F", "SILVER": "SI=F",
    "COBRE": "HG=F", "COPPER": "HG=F", "PETROLEO": "CL=F", "OIL": "CL=F",
    "WTI": "CL=F", "BRENT": "BZ=F", "GAS": "NG=F", "NATGAS": "NG=F",
    "MILHO": "ZC=F", "CORN": "ZC=F", "TRIGO": "ZW=F", "WHEAT": "ZW=F",
    "SOJA": "ZS=F", "CAFE": "KC=F", "COFFEE": "KC=F", "ACUCAR": "SB=F",
    "SUGAR": "SB=F", "CACAU": "CC=F", "ALGODAO": "CT=F",
    "SPX": "^GSPC", "SP500": "^GSPC", "DJI": "^DJI", "DOW": "^DJI",
    "NASDAQ": "^IXIC", "NDX": "^NDX", "RUSSELL": "^RUT", "VIX": "^VIX",
    "IBOV": "^BVSP", "IBOVESPA": "^BVSP", "DAX": "^GDAXI", "NIKKEI": "^N225",
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X", "USDBRL": "BRL=X", "DOLAR": "BRL=X",
    "EURBRL": "EURBRL=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "USDMXN": "MXN=X", "USDCNY": "CNY=X",
}

_LABELS = {yf: lb for uni in UNIVERSES.values() for yf, lb in uni}


def resolve(symbol: str) -> str:
    """Alias amigável -> ticker yfinance; tickers já válidos passam direto."""
    s = (symbol or "").strip().upper()
    return ALIASES.get(s, s)


def label_for(yf_sym: str) -> str:
    return _LABELS.get(yf_sym, yf_sym)


# ── quotes (watch / alertas) ─────────────────────────────────────────────

def _quote_one(yf_sym):
    import yfinance as yf
    fi = yf.Ticker(yf_sym).fast_info
    last = _f(fi["last_price"])
    prev = _f(fi["previous_close"])
    return {
        "base": yf_sym,
        "symbol": yf_sym,
        "label": label_for(yf_sym),
        "market": "tradfi",
        "last": last,
        "pct24h": (last / prev - 1) * 100 if last and prev else None,
        "high24": _f(fi["day_high"]),
        "low24": _f(fi["day_low"]),
        "vol_usd": _f(fi["last_volume"]) or None,   # p/ tradfi é volume em unid.
        "funding": None,
        "next_funding_ts": None,
    }


def quotes(symbols: list[str]) -> dict:
    """{símbolo original: row} — fast_info em paralelo, cache 15s por símbolo."""
    out = {}

    def one(orig):
        yf_sym = resolve(orig)
        try:
            row = _cached(("tq", yf_sym), 15, lambda: _quote_one(yf_sym))
            return orig, {**row, "base": orig.upper()}
        except Exception:
            return orig, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for orig, row in pool.map(one, symbols):
            if row:
                out[orig] = row
    return out


# ── sparkline ────────────────────────────────────────────────────────────

def closes(symbol: str, tf: str = "15m", bars: int = 96) -> list:
    yf_sym = resolve(symbol)
    interval = tf if tf in ("1m", "5m", "15m", "30m", "1h", "1d") else "15m"
    period = "5d" if interval != "1d" else "6mo"

    def fetch():
        import yfinance as yf
        df = yf.Ticker(yf_sym).history(period=period, interval=interval)
        return [_f(v) for v in df["Close"].tail(bars).tolist()]

    return _cached(("tspark", yf_sym, interval, bars), 600, fetch)


# ── screener por universo ────────────────────────────────────────────────

def screener_rows(market: str) -> list:
    uni = UNIVERSES.get(market)
    if not uni:
        raise ValueError(f"market deve ser um de {list(UNIVERSES)}")

    def fetch():
        import yfinance as yf
        tickers = [yf_sym for yf_sym, _ in uni]
        df = yf.download(tickers, period="60d", interval="1d",
                         group_by="ticker", progress=False, threads=True,
                         auto_adjust=True)
        rows = []
        for yf_sym, lb in uni:
            try:
                sub = df[yf_sym].dropna(how="all") if len(tickers) > 1 else df
                c = sub["Close"].dropna()
                if len(c) < 2:
                    continue
                last = float(c.iloc[-1])
                def ret(n):
                    return (last / float(c.iloc[-n - 1]) - 1) * 100 if len(c) > n else None
                h, lo, cl = sub["High"], sub["Low"], sub["Close"]
                trs = []
                for i in range(1, len(sub)):
                    if any(v != v for v in (h.iloc[i], lo.iloc[i], cl.iloc[i - 1])):
                        continue
                    trs.append(max(h.iloc[i] - lo.iloc[i],
                                   abs(h.iloc[i] - cl.iloc[i - 1]),
                                   abs(lo.iloc[i] - cl.iloc[i - 1])))
                atr = sum(trs[-14:]) / min(14, len(trs)) if trs else None
                vol = _f(sub["Volume"].iloc[-1]) if "Volume" in sub else None
                rows.append({
                    "base": lb if market != "stocks" else yf_sym,
                    "symbol": yf_sym,
                    "label": lb,
                    "market": "tradfi",
                    "last": last,
                    "pct24h": ret(1), "ret7d": ret(5), "ret30d": ret(21),
                    "atr_pct": atr / last * 100 if atr and last else None,
                    "vol_usd": vol * last if vol else None,
                    "funding": None,
                })
            except Exception:
                continue
        rows.sort(key=lambda r: r.get("vol_usd") or 0, reverse=True)
        return rows

    return _cached(("tscreener", market), 1800, fetch)


# ── DES (descrição do instrumento) ───────────────────────────────────────

_INFO_FIELDS = {
    "name": ("longName", "shortName"),
    "sector": ("sector",), "industry": ("industry",),
    "market_cap": ("marketCap",), "pe": ("trailingPE",),
    "forward_pe": ("forwardPE",), "eps": ("trailingEps",),
    "dividend_yield": ("dividendYield",), "beta": ("beta",),
    "week52_high": ("fiftyTwoWeekHigh",), "week52_low": ("fiftyTwoWeekLow",),
    "avg_volume": ("averageVolume",), "currency": ("currency",),
    "exchange_name": ("fullExchangeName", "exchange"),
    "summary": ("longBusinessSummary",), "website": ("website",),
}


def describe(symbol: str) -> dict:
    yf_sym = resolve(symbol)

    def fetch():
        import yfinance as yf
        t = yf.Ticker(yf_sym)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out = {"kind": "tradfi", "base": symbol.upper(), "symbol": yf_sym,
               "label": label_for(yf_sym), "market": "tradfi"}
        for key, srcs in _INFO_FIELDS.items():
            v = next((info[s] for s in srcs if info.get(s) is not None), None)
            if key in ("name", "sector", "industry", "currency",
                       "exchange_name", "summary", "website"):
                out[key] = str(v)[:600] if v is not None else None
            else:
                out[key] = _f(v)

        fi = t.fast_info
        last = _f(fi["last_price"])
        prev = _f(fi["previous_close"])
        out.update({
            "last": last,
            "pct24h": (last / prev - 1) * 100 if last and prev else None,
            "high24": _f(fi["day_high"]), "low24": _f(fi["day_low"]),
            "vol_usd": _f(fi["last_volume"]),
        })

        hist = t.history(period="6mo", interval="1d")
        c = hist["Close"].dropna()
        if len(c) >= 2:
            lastc = float(c.iloc[-1])
            def ret(n):
                return (lastc / float(c.iloc[-n - 1]) - 1) * 100 if len(c) > n else None
            out["ret7d"], out["ret30d"] = ret(5), ret(21)
            h, lo, cl = hist["High"], hist["Low"], hist["Close"]
            trs = []
            for i in range(max(1, len(hist) - 14), len(hist)):
                vals = (h.iloc[i], lo.iloc[i], cl.iloc[i - 1])
                if any(v != v for v in vals):
                    continue
                trs.append(max(h.iloc[i] - lo.iloc[i],
                               abs(h.iloc[i] - cl.iloc[i - 1]),
                               abs(lo.iloc[i] - cl.iloc[i - 1])))
            out["atr_pct"] = (sum(trs) / len(trs)) / lastc * 100 if trs and lastc else None
            out["price_hist"] = {
                "dates": [int(ts.timestamp() * 1000) for ts in c.index],
                "closes": [float(v) for v in c.tolist()],
            }
        return out

    return _cached(("tdes", yf_sym), 900, fetch)


def is_tradfi_symbol(symbol: str) -> bool:
    """Heurística p/ modo auto: aliases e formatos yfinance (=F/=X/^/.)"""
    s = (symbol or "").strip().upper()
    return s in ALIASES or "=" in s or s.startswith("^") or "." in s
