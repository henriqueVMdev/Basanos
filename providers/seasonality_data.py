"""Sazonalidade de qualquer ativo compatível com Yahoo Finance."""
from __future__ import annotations

import time

from common import to_float, ttl_cache

_cached = ttl_cache()
WINDOWS = (20, 15, 10, 5, 2)


def _clean(v, digits=4):
    x = to_float(v)
    return None if x is None else round(x, digits)


def _path(returns, years):
    """Curva média acumulada (%) por dia do ano, na janela de N anos."""
    import pandas as pd

    r = returns[returns.index >= returns.index.max() - pd.DateOffset(years=years)]
    # Chave mês-dia mantém mar-dez alinhados entre anos bissextos e comuns.
    keys = r.index.strftime("%m-%d")
    canonical = pd.date_range("2001-01-01", "2001-12-31").strftime("%m-%d")
    daily = r.groupby(keys).mean().reindex(canonical, fill_value=0).fillna(0)
    return [_clean(x) for x in (((1 + daily).cumprod() - 1) * 100)]


def analyze(symbol):
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol obrigatório")
    return _cached(sym, 3600, lambda: _analyze(sym))


def _analyze(sym):
    import pandas as pd
    import yfinance as yf

    try:
        from providers import tradfi_data
        yf_sym = tradfi_data.resolve(sym)
    except Exception:
        yf_sym = sym

    ticker = yf.Ticker(yf_sym)
    hist = ticker.history(period="max", interval="1d", auto_adjust=True)
    if hist is None or hist.empty or len(hist) < 365:
        raise ValueError(f"histórico insuficiente para {sym}")

    close = hist["Close"].dropna()
    close.index = pd.to_datetime(close.index, utc=True).tz_convert(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    returns = close.pct_change().dropna()

    monthly = close.resample("ME").last().pct_change() * 100
    if len(monthly):
        monthly = monthly.iloc[:-1]  # mês corrente ainda aberto enviesaria as estatísticas

    heatmap = []
    for year in sorted(set(close.index.year))[-20:]:
        vals = {int(d.month): _clean(v, 2) for d, v in monthly[monthly.index.year == year].items()}
        heatmap.append({"year": int(year), "months": [vals.get(m) for m in range(1, 13)]})

    mr = monthly[monthly.index >= monthly.index.max() - pd.DateOffset(years=20)]
    stats = []
    for m in range(1, 13):
        x = mr[mr.index.month == m].dropna()
        stats.append({"month": m, "avg_pct": _clean(x.mean(), 2), "median_pct": _clean(x.median(), 2),
                      "win_rate_pct": _clean((x > 0).mean() * 100, 1),
                      "stdev_pct": _clean(x.std(), 2), "samples": len(x)})

    intra = _intra_month(close)
    ranked = sorted((x for x in stats if x["avg_pct"] is not None), key=lambda x: x["avg_pct"])
    try:
        name = ticker.info.get("shortName") or sym
    except Exception:
        name = sym

    return {
        "symbol": sym, "yf_symbol": yf_sym, "name": name,
        "history": {"start": str(close.index.min().date()), "end": str(close.index.max().date()),
                    "observations": len(close), "years": len(set(close.index.year))},
        "day_of_year": list(range(1, 366)),
        "paths": {str(w): _path(returns, w) for w in WINDOWS},
        "heatmap": heatmap, "monthly_stats": stats,
        "intra_month": {"month": int(close.index.max().month), "points": intra},
        "best_months": list(reversed(ranked[-3:])), "worst_months": ranked[:3],
        "methodology": "Retornos ajustados; média sazonal por dia do ano. "
                       "Heatmap e estatísticas mensais até 20 anos.",
        "source": "Yahoo Finance via yfinance", "ts": int(time.time() * 1000),
    }


def _intra_month(close):
    """Trajetória média dentro do mês corrente, por dia útil, nos últimos 20 anos."""
    import pandas as pd

    month = close.index.max().month
    subset = close[(close.index >= close.index.max() - pd.DateOffset(years=20))
                   & (close.index.month == month)]
    series = []
    for _, x in subset.groupby(subset.index.year):
        if len(x) >= 3:
            z = (x / x.iloc[0] - 1) * 100
            z.index = range(1, len(z) + 1)
            series.append(z)
    if not series:
        return []

    panel = pd.concat(series, axis=1)
    out = []
    for day, row in panel.iterrows():
        x = row.dropna()
        out.append({"trading_day": int(day), "avg_pct": _clean(x.mean()),
                    "low_pct": _clean(x.quantile(.25)), "high_pct": _clean(x.quantile(.75)),
                    "samples": len(x)})
    return out
