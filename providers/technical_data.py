"""
GT — análise técnica multi-ativo e multi-mercado:
histórico unificado (cripto via ccxt, tradicional via yfinance) em qualquer
timeframe (intraday/diário/semanal/mensal), indicadores (SMA/EMA/RSI/MACD/
Bollinger), volume financeiro, comparação entre ativos (retornos acumulados),
spread charts (diferença e razão), correlação (rolling e matriz) e estudos
customizados via expressão (ex.: "EMA(close,9) - EMA(close,21)").

Comparação entre contratos futuros = mesmo pipeline com tickers de contratos
individuais do Yahoo (CLZ26.NYM etc., listados por commodities_data).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import to_float, ttl_cache

_cached = ttl_cache()


# histórico unificado
# tradfi: intervalo -> (interval yahoo, period máximo p/ esse intervalo)
_YF_INTERVALS = {
    "15m": ("15m", "60d"), "30m": ("30m", "60d"), "1h": ("1h", "730d"),
    "4h": ("1h", "730d"), "1d": ("1d", "10y"), "1wk": ("1wk", "max"),
    "1mo": ("1mo", "max"),
}

_TTL = {"15m": 300, "30m": 300, "1h": 600, "4h": 600,
        "1d": 1800, "1wk": 3600, "1mo": 3600}


def history(symbol: str, market: str, interval: str, bars: int,
            exchange: str = "bybit") -> pd.DataFrame:
    """DataFrame [Open, High, Low, Close, Volume], índice datetime UTC naive."""
    bars = min(int(bars), 3000)

    def fetch():
        if market == "crypto":
            from providers.market_data import fetch_ohlcv
            tf = interval if interval != "1wk" else "1w"
            tf = tf if tf != "1mo" else "1M"
            df = fetch_ohlcv(symbol, timeframe=tf, exchange=exchange,
                             limit=1000, total=bars if bars > 1000 else None)
            return df.tail(bars)
        from providers import tradfi_data
        import yfinance as yf
        yf_int, period = _YF_INTERVALS.get(interval, ("1d", "10y"))
        df = yf.Ticker(tradfi_data.resolve(symbol)).history(
            period=period, interval=yf_int)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        if interval == "4h":
            df = df.resample("4h").agg({"Open": "first", "High": "max",
                                        "Low": "min", "Close": "last",
                                        "Volume": "sum"}).dropna(how="all")
        return df.tail(bars)

    key = ("hist", market, symbol.upper(), interval, bars, exchange)
    return _cached(key, _TTL.get(interval, 600), fetch)


# indicadores
def _sma(s, n):
    return s.rolling(int(n)).mean()


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False).mean()


def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(s, fast=12, slow=26, signal=9):
    line = _ema(s, fast) - _ema(s, slow)
    sig = _ema(line, signal)
    return line, sig, line - sig


def _bollinger(s, n=20, k=2.0):
    mid = _sma(s, n)
    sd = s.rolling(int(n)).std()
    return mid, mid + k * sd, mid - k * sd


# estudos customizados (expressão com whitelist)
_EXPR_ALLOWED = re.compile(r"^[A-Za-z0-9_+\-*/(),.\s]+$")
_EXPR_NAMES = {"open", "high", "low", "close", "volume",
               "SMA", "EMA", "RSI", "STD", "MAX", "MIN", "ROC",
               "SHIFT", "ABS", "LOG", "DIFF"}


def eval_custom(expr: str, df: pd.DataFrame) -> pd.Series:
    if not expr or len(expr) > 300 or not _EXPR_ALLOWED.match(expr):
        raise ValueError("expressão inválida (só nomes, números e + - * / ( ) ,)")
    for name in re.findall(r"[A-Za-z_]\w*", expr):
        if name not in _EXPR_NAMES:
            raise ValueError(
                f"'{name}' não permitido. Use: {', '.join(sorted(_EXPR_NAMES))}")
    env = {
        "open": df["Open"], "high": df["High"], "low": df["Low"],
        "close": df["Close"], "volume": df["Volume"],
        "SMA": _sma, "EMA": _ema, "RSI": _rsi,
        "STD": lambda s, n: s.rolling(int(n)).std(),
        "MAX": lambda s, n: s.rolling(int(n)).max(),
        "MIN": lambda s, n: s.rolling(int(n)).min(),
        "ROC": lambda s, n: s.pct_change(int(n)) * 100,
        "SHIFT": lambda s, n: s.shift(int(n)),
        "ABS": lambda s: s.abs(),
        "LOG": lambda s: np.log(s.replace(0, np.nan)),
        "DIFF": lambda s: s.diff(),
    }
    out = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 — whitelist acima
    if not isinstance(out, pd.Series):
        raise ValueError("a expressão deve produzir uma série (não um número)")
    return out


# payloads
def _ser(s: pd.Series) -> list:
    return [None if (x := to_float(v)) is None else round(x, 6) for v in s]


def _ts(df: pd.DataFrame) -> list:
    return [int(t.timestamp() * 1000) for t in df.index]


def single_chart(sym: dict, interval: str, bars: int, studies: list) -> dict:
    df = history(sym["s"], sym.get("market", "crypto"), interval, bars,
                 sym.get("exchange", "bybit"))
    if df.empty:
        return {"error": f"sem dados para {sym['s']} em {interval}"}
    close = df["Close"]
    out = {
        "symbol": sym["s"].upper(), "interval": interval,
        "ts": _ts(df),
        "open": _ser(df["Open"]), "high": _ser(df["High"]),
        "low": _ser(df["Low"]), "close": _ser(close),
        "volume": _ser(df["Volume"]),
        "vol_fin": _ser(df["Volume"] * close),   # volume financeiro
        "overlays": [], "panels": [],
    }
    for st in studies or []:
        kind = st.get("type")
        n = int(st.get("n") or 14)
        try:
            if kind == "sma":
                out["overlays"].append({"name": f"SMA{n}", "values": _ser(_sma(close, n))})
            elif kind == "ema":
                out["overlays"].append({"name": f"EMA{n}", "values": _ser(_ema(close, n))})
            elif kind == "bb":
                k = float(st.get("k") or 2)
                mid, up, lo = _bollinger(close, n, k)
                out["overlays"] += [
                    {"name": f"BB{n} sup", "values": _ser(up), "dash": "dot"},
                    {"name": f"BB{n} méd", "values": _ser(mid), "dash": "dot"},
                    {"name": f"BB{n} inf", "values": _ser(lo), "dash": "dot"},
                ]
            elif kind == "rsi":
                out["panels"].append({"name": f"RSI({n})", "kind": "rsi",
                                      "series": [{"name": "RSI", "values": _ser(_rsi(close, n))}]})
            elif kind == "macd":
                line, sig, hist = _macd(close)
                out["panels"].append({"name": "MACD(12,26,9)", "kind": "macd",
                                      "series": [
                                          {"name": "MACD", "values": _ser(line)},
                                          {"name": "Sinal", "values": _ser(sig)},
                                          {"name": "Hist", "values": _ser(hist), "bar": True}]})
            elif kind == "custom":
                expr = st.get("expr") or ""
                out["panels"].append({"name": st.get("name") or expr[:40], "kind": "custom",
                                      "series": [{"name": expr[:40],
                                                  "values": _ser(eval_custom(expr, df))}]})
        except ValueError as e:
            out.setdefault("study_errors", []).append(str(e))
    return out


def compare_chart(symbols: list, interval: str, bars: int,
                  corr_window: int = 30) -> dict:
    """Retornos acumulados alinhados + correlação (matriz e, p/ 2 ativos,
    rolling) + spread (diferença e razão) quando são exatamente 2."""
    closes = {}
    for sym in symbols[:8]:
        df = history(sym["s"], sym.get("market", "crypto"), interval, bars,
                     sym.get("exchange", "bybit"))
        if not df.empty:
            s = df["Close"]
            # d/semana/mês: normaliza p/ data — Yahoo carimba 04:00/05:00 UTC
            # (tz da bolsa) e cripto 00:00; sem isso o inner-join não casa
            if interval in ("1d", "1wk", "1mo"):
                s = s.copy()
                s.index = s.index.normalize()
                s = s[~s.index.duplicated(keep="last")]
            closes[sym["s"].upper()] = s
    if len(closes) < 2:
        return {"error": "preciso de pelo menos 2 ativos com dados"}

    aligned = pd.DataFrame(closes).dropna()
    if len(aligned) < 10:
        return {"error": "menos de 10 candles em comum — tente timeframe diário"}

    rets = aligned.pct_change().dropna()
    cum = (1 + rets).cumprod() - 1

    out = {
        "interval": interval,
        "ts": [int(t.timestamp() * 1000) for t in cum.index],
        "names": list(aligned.columns),
        "cumret": {c: _ser(cum[c] * 100) for c in cum.columns},
        "corr_matrix": {
            "labels": list(rets.columns),
            "matrix": [[round(float(v), 3) for v in row]
                       for row in rets.corr().values],
        },
        "n_common": len(aligned),
    }

    if len(aligned.columns) == 2:
        a, b = aligned.columns[0], aligned.columns[1]
        out["spread"] = {
            "pair": f"{a} - {b}",
            "ts": [int(t.timestamp() * 1000) for t in aligned.index],
            "diff": _ser(aligned[a] - aligned[b]),
            "ratio": _ser(aligned[a] / aligned[b]),
        }
        w = max(10, int(corr_window))
        roll = rets[a].rolling(w).corr(rets[b])
        out["roll_corr"] = {"window": w, "values": _ser(roll)}
    return out
