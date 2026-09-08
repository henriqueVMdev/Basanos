"""
EQS — screening fundamentalista estilo Bloomberg via Yahoo screener
(yf.screen + EquityQuery/FundQuery). O filtro roda server-side no Yahoo
sobre o mercado inteiro (~40 países), não sobre uma lista fixa.

Métricas de query (múltiplos, margens, dívida, crescimento) filtram mas nem
todas voltam no quote; a tabela mostra os campos retornados + score relativo
calculado aqui (percentis dentro do conjunto filtrado = oportunidade relativa
dentro do peer group que o usuário definiu por setor/país).
"""

from __future__ import annotations

import time

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
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ── metadados p/ a UI ────────────────────────────────────────────────────

REGIONS = [
    ("us", "Estados Unidos"), ("br", "Brasil"), ("gb", "Reino Unido"),
    ("de", "Alemanha"), ("fr", "França"), ("jp", "Japão"), ("cn", "China"),
    ("hk", "Hong Kong"), ("in", "Índia"), ("ca", "Canadá"), ("au", "Austrália"),
    ("kr", "Coreia do Sul"), ("ch", "Suíça"), ("nl", "Holanda"), ("es", "Espanha"),
    ("it", "Itália"), ("se", "Suécia"), ("mx", "México"), ("ar", "Argentina"),
    ("cl", "Chile"), ("sa", "Arábia Saudita"), ("sg", "Singapura"), ("tw", "Taiwan"),
]

SECTORS = [
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive", "Energy",
    "Basic Materials", "Real Estate", "Utilities",
]

# métrica amigável -> (campo yahoo, rótulo, unidade)
METRICS = {
    "pe":              ("peratio.lasttwelvemonths", "P/E (12m)", "x"),
    "pb":              ("pricebookratio.quarterly", "P/B", "x"),
    "peg":             ("pegratio_5y", "PEG 5a", "x"),
    "ev_ebitda":       ("lastclosetevebitda.lasttwelvemonths", "EV/EBITDA", "x"),
    "ps":              ("lastclosemarketcaptotalrevenue.lasttwelvemonths", "P/S", "x"),
    "mcap":            ("intradaymarketcap", "Market cap", "$"),
    "div_yield":       ("forward_dividend_yield", "Div. yield", "%"),
    "rev_growth":      ("totalrevenues1yrgrowth.lasttwelvemonths", "Cresc. receita 1a", "%"),
    "eps_growth":      ("epsgrowth.lasttwelvemonths", "Cresc. EPS", "%"),
    "net_margin":      ("netincomemargin.lasttwelvemonths", "Margem líquida", "%"),
    "gross_margin":    ("grossprofitmargin.lasttwelvemonths", "Margem bruta", "%"),
    "ebitda_margin":   ("ebitdamargin.lasttwelvemonths", "Margem EBITDA", "%"),
    "debt_equity":     ("totaldebtequity.lasttwelvemonths", "Dívida/PL", "%"),
    "net_debt_ebitda": ("netdebtebitda.lasttwelvemonths", "Dív.líq/EBITDA", "x"),
    "roe":             ("returnonequity.lasttwelvemonths", "ROE", "%"),
    "roa":             ("returnonassets.lasttwelvemonths", "ROA", "%"),
    "beta":            ("beta", "Beta", "x"),
    "avg_vol":         ("avgdailyvol3m", "Vol. médio 3m", "ações"),
}

SORT_FIELDS = {
    "mcap": "intradaymarketcap",
    "pe": "peratio.lasttwelvemonths",
    "div_yield": "forward_dividend_yield",
    "avg_vol": "avgdailyvol3m",
    "pct_change": "percentchange",
}

# screens prontos de fundos/ETFs/bonds (Yahoo predefined)
FUND_SCREENS = [
    ("top_etfs_us", "Top ETFs (EUA)"),
    ("top_performing_etfs", "ETFs melhor desempenho"),
    ("technology_etfs", "ETFs de tecnologia"),
    ("bond_etfs", "ETFs de bonds"),
    ("high_yield_bond", "Bonds high yield"),
    ("top_mutual_funds", "Top fundos mútuos"),
    ("solid_large_growth_funds", "Fundos large growth"),
    ("solid_midcap_growth_funds", "Fundos midcap growth"),
    ("conservative_foreign_funds", "Fundos internacionais conservadores"),
]

# bolsas OTC (pink sheets) — excluídas por padrão
_OTC_EXCHANGES = {"PNK", "OTC", "OQB", "OQX", "OEM", "OGM"}


def meta():
    return {
        "regions": [{"key": k, "label": v} for k, v in REGIONS],
        "sectors": SECTORS,
        "metrics": [{"key": k, "label": lb, "unit": u}
                    for k, (_, lb, u) in METRICS.items()],
        "sorts": list(SORT_FIELDS),
        "fund_screens": [{"key": k, "label": v} for k, v in FUND_SCREENS],
    }


# ── screening de ações ───────────────────────────────────────────────────

def _build_query(filters: dict):
    import yfinance as yf
    EQ = yf.EquityQuery
    nodes = []

    regions = [r for r in (filters.get("regions") or []) if r]
    if regions:
        subs = [EQ("eq", ["region", r]) for r in regions]
        nodes.append(subs[0] if len(subs) == 1 else EQ("or", subs))

    sectors = [s for s in (filters.get("sectors") or []) if s]
    if sectors:
        subs = [EQ("eq", ["sector", s]) for s in sectors]
        nodes.append(subs[0] if len(subs) == 1 else EQ("or", subs))

    for key, spec in (filters.get("metrics") or {}).items():
        if key not in METRICS or not isinstance(spec, dict):
            continue
        field = METRICS[key][0]
        lo, hi = _f(spec.get("min")), _f(spec.get("max"))
        if lo is not None and hi is not None:
            nodes.append(EQ("btwn", [field, lo, hi]))
        elif lo is not None:
            nodes.append(EQ("gt", [field, lo]))
        elif hi is not None:
            nodes.append(EQ("lt", [field, hi]))

    if not nodes:
        nodes.append(EQ("eq", ["region", "us"]))  # query vazia não é aceita
    return nodes[0] if len(nodes) == 1 else EQ("and", nodes)


def _quote_row(q: dict) -> dict:
    return {
        "symbol": q.get("symbol"),
        "name": (q.get("shortName") or q.get("longName") or "")[:40],
        "exchange": q.get("exchange"),
        "last": _f(q.get("regularMarketPrice")),
        "pct_change": _f(q.get("regularMarketChangePercent")),
        "mcap": _f(q.get("marketCap")),
        "pe": _f(q.get("trailingPE")),
        "forward_pe": _f(q.get("forwardPE")),
        "pb": _f(q.get("priceToBook")),
        "eps": _f(q.get("epsTrailingTwelveMonths")),
        "div_yield": _f(q.get("dividendYield")),
        "avg_vol": _f(q.get("averageDailyVolume3Month")),
        "chg_52w": _f(q.get("fiftyTwoWeekChangePercent")),
        "currency": q.get("currency"),
        "quote_type": q.get("quoteType"),
    }


def _pct_ranks(values: list) -> list:
    """Percentil (0-1) de cada valor dentro da lista; None preservado."""
    known = sorted(v for v in values if v is not None)
    n = len(known)
    if n < 2:
        return [None] * len(values)
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            below = sum(1 for k in known if k < v)
            out.append(below / (n - 1))
    return out


def _add_scores(rows: list) -> None:
    """Score relativo 0-100 dentro do conjunto filtrado (peer group):
    valor (P/E e P/B baixos), dividendo (yield alto), momento (52s alto)."""
    inv = lambda v: 1 / v if v and v > 0 else None
    val_pe = _pct_ranks([inv(r["pe"]) for r in rows])
    val_pb = _pct_ranks([inv(r["pb"]) for r in rows])
    div = _pct_ranks([r["div_yield"] for r in rows])
    mom = _pct_ranks([r["chg_52w"] for r in rows])
    for i, r in enumerate(rows):
        vals = [x for x in (val_pe[i], val_pb[i]) if x is not None]
        comp = {
            "score_valor": sum(vals) / len(vals) * 100 if vals else None,
            "score_div": div[i] * 100 if div[i] is not None else None,
            "score_momento": mom[i] * 100 if mom[i] is not None else None,
        }
        avail = [v for v in comp.values() if v is not None]
        comp["score"] = sum(avail) / len(avail) if avail else None
        r.update({k: (round(v, 1) if v is not None else None)
                  for k, v in comp.items()})


def run_equity_screen(filters: dict) -> dict:
    import json
    key = ("eqs", json.dumps(filters, sort_keys=True))

    def fetch():
        import yfinance as yf
        q = _build_query(filters)
        sort = SORT_FIELDS.get(filters.get("sort") or "mcap", "intradaymarketcap")
        size = min(int(filters.get("size") or 100), 250)
        resp = yf.screen(q, sortField=sort,
                         sortAsc=bool(filters.get("sort_asc")), size=size)
        quotes = resp.get("quotes") or []
        rows = [_quote_row(x) for x in quotes]
        if not filters.get("include_otc"):
            rows = [r for r in rows if r["exchange"] not in _OTC_EXCHANGES]
        _add_scores(rows)
        rows.sort(key=lambda r: r.get("score") or -1, reverse=True)
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return {"rows": rows, "total_matches": resp.get("total"),
                "ts": int(time.time() * 1000)}

    return _cached(key, 300, fetch)


# ── screening de fundos / ETFs / bonds (screens prontos) ─────────────────

def run_fund_screen(screen_key: str) -> dict:
    if screen_key not in {k for k, _ in FUND_SCREENS}:
        raise ValueError(f"screen deve ser um de {[k for k, _ in FUND_SCREENS]}")

    def fetch():
        import yfinance as yf
        resp = yf.screen(screen_key, size=100)
        rows = []
        for q in resp.get("quotes") or []:
            rows.append({
                "symbol": q.get("symbol"),
                "name": (q.get("shortName") or q.get("longName") or "")[:44],
                "quote_type": q.get("quoteType"),
                "exchange": q.get("exchange"),
                "last": _f(q.get("regularMarketPrice")),
                "pct_change": _f(q.get("regularMarketChangePercent")),
                "ytd": _f(q.get("ytdReturn")),
                "chg_52w": _f(q.get("fiftyTwoWeekChangePercent")),
                "net_assets": _f(q.get("netAssets")),
                "div_yield": _f(q.get("trailingAnnualDividendYield")),
                "expense_ratio": _f(q.get("netExpenseRatio")),
                "avg_vol": _f(q.get("averageDailyVolume3Month")),
            })
        return {"rows": rows, "total_matches": resp.get("total"),
                "ts": int(time.time() * 1000)}

    return _cached(("fund_screen", screen_key), 900, fetch)
