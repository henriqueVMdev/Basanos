"""
EA/FA — análise completa de uma empresa listada (estilo Bloomberg FA):
demonstrações financeiras, múltiplos, estimativas e recomendações de
analistas, preço-alvo, calendário de balanços, participação acionária,
insider transactions, dividendos e recompras, comparação com pares.

Fonte: Yahoo via yfinance (sem chave). Payload único cacheado 15 min —
a primeira carga faz ~8 chamadas e leva alguns segundos.
"""

from __future__ import annotations

import time

from common import ttl_cache, to_float

_cached = ttl_cache()


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# linhas-chave das demonstrações (nome yahoo -> chave da API)
_INCOME_ROWS = [
    ("Total Revenue", "receita"),
    ("Gross Profit", "lucro_bruto"),
    ("EBITDA", "ebitda"),
    ("Operating Income", "lucro_operacional"),
    ("Net Income", "lucro_liquido"),
    ("Basic EPS", "eps"),
]
_BALANCE_ROWS = [
    ("Total Assets", "ativos"),
    ("Total Debt", "divida_total"),
    ("Cash And Cash Equivalents", "caixa"),
    ("Stockholders Equity", "patrimonio"),
]
_CASHFLOW_ROWS = [
    ("Operating Cash Flow", "fco"),
    ("Capital Expenditure", "capex"),
    ("Free Cash Flow", "fcl"),
    ("Repurchase Of Capital Stock", "recompras"),
    ("Cash Dividends Paid", "dividendos_pagos"),
]


def _extract(df, rows, n=5):
    """DataFrame yahoo (linhas x períodos) -> {periods: [...], data: {chave: [...]}}"""
    if df is None or getattr(df, "empty", True):
        return None
    cols = list(df.columns)[:n]
    out = {"periods": [c.strftime("%Y-%m") for c in cols], "data": {}}
    for yh_name, key in rows:
        if yh_name in df.index:
            vals = [to_float(df.loc[yh_name, c]) for c in cols]
            if any(v is not None for v in vals):
                out["data"][key] = vals
    # margens calculadas (income)
    rec = out["data"].get("receita")
    if rec:
        for src, mkey in (("lucro_bruto", "margem_bruta"),
                          ("ebitda", "margem_ebitda"),
                          ("lucro_liquido", "margem_liquida")):
            num = out["data"].get(src)
            if num:
                out["data"][mkey] = [
                    round(n_ / r * 100, 2) if n_ is not None and r else None
                    for n_, r in zip(num, rec)]
    return out


_SUFFIX_REGION = {".SA": "br", ".L": "gb", ".DE": "de", ".PA": "fr",
                  ".T": "jp", ".TO": "ca", ".AX": "au", ".HK": "hk",
                  ".KS": "kr", ".SW": "ch", ".AS": "nl", ".MC": "es",
                  ".MI": "it", ".ST": "se", ".MX": "mx", ".NS": "in"}


def _peers(yf_sym: str, sector: str | None, mcap) -> list:
    """Top do mesmo setor/região por market cap (via Yahoo screener)."""
    if not sector:
        return []
    import yfinance as yf
    region = "us"
    for suf, reg in _SUFFIX_REGION.items():
        if yf_sym.upper().endswith(suf):
            region = reg
            break
    EQ = yf.EquityQuery
    q = EQ("and", [EQ("eq", ["region", region]), EQ("eq", ["sector", sector])])
    resp = yf.screen(q, sortField="intradaymarketcap", sortAsc=False, size=12)
    peers = []
    for x in resp.get("quotes") or []:
        if x.get("symbol", "").upper() == yf_sym.upper():
            continue
        peers.append({
            "symbol": x.get("symbol"),
            "name": (x.get("shortName") or "")[:28],
            "mcap": to_float(x.get("marketCap")),
            "pe": to_float(x.get("trailingPE")),
            "forward_pe": to_float(x.get("forwardPE")),
            "pb": to_float(x.get("priceToBook")),
            "div_yield": to_float(x.get("dividendYield")),
            "chg_52w": to_float(x.get("fiftyTwoWeekChangePercent")),
        })
        if len(peers) >= 8:
            break
    return peers


def analyze(symbol: str) -> dict:
    from providers import insider_data, tradfi_data
    yf_sym = tradfi_data.resolve(symbol)

    def fetch():
        import yfinance as yf
        t = yf.Ticker(yf_sym)
        info = _safe(lambda: t.info, {}) or {}

        out = {
            "symbol": symbol.upper(), "yf_symbol": yf_sym,
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"), "industry": info.get("industry"),
            "country": info.get("country"), "currency": info.get("currency"),
            "exchange_name": info.get("fullExchangeName"),
            "summary": (info.get("longBusinessSummary") or "")[:500] or None,
            "last": to_float(info.get("currentPrice") or info.get("regularMarketPrice")),
            "pct24h": to_float(info.get("regularMarketChangePercent")),
            "mcap": to_float(info.get("marketCap")),
            "multiples": {
                "pe": to_float(info.get("trailingPE")),
                "forward_pe": to_float(info.get("forwardPE")),
                "ev_ebitda": to_float(info.get("enterpriseToEbitda")),
                "pb": to_float(info.get("priceToBook")),
                "ps": to_float(info.get("priceToSalesTrailing12Months")),
                "peg": to_float(info.get("trailingPegRatio")),
                "div_yield": to_float(info.get("dividendYield")),
                "payout": to_float(info.get("payoutRatio")),
                "roe": to_float(info.get("returnOnEquity")),
                "beta": to_float(info.get("beta")),
            },
            "ts": int(time.time() * 1000),
        }

        # demonstrações — anual e trimestral
        out["statements"] = {
            "income_a": _extract(_safe(lambda: t.income_stmt), _INCOME_ROWS),
            "income_q": _extract(_safe(lambda: t.quarterly_income_stmt), _INCOME_ROWS),
            "balance_a": _extract(_safe(lambda: t.balance_sheet), _BALANCE_ROWS),
            "cashflow_a": _extract(_safe(lambda: t.cashflow), _CASHFLOW_ROWS),
        }

        # analistas: preço-alvo + recomendações
        out["price_targets"] = _safe(lambda: {
            k: to_float(v) for k, v in (t.analyst_price_targets or {}).items()})
        recs = _safe(lambda: t.recommendations_summary)
        if recs is not None and not recs.empty:
            r0 = recs.iloc[0]
            out["recommendations"] = {
                "strong_buy": int(r0.get("strongBuy") or 0),
                "buy": int(r0.get("buy") or 0),
                "hold": int(r0.get("hold") or 0),
                "sell": int(r0.get("sell") or 0),
                "strong_sell": int(r0.get("strongSell") or 0),
            }

        # calendário de balanços + estimativas do próximo tri
        cal = _safe(lambda: t.calendar, {}) or {}
        ed = cal.get("Earnings Date")
        out["calendar"] = {
            "next_earnings": str(ed[0]) if isinstance(ed, list) and ed else None,
            "eps_est": to_float(cal.get("Earnings Average")),
            "eps_low": to_float(cal.get("Earnings Low")),
            "eps_high": to_float(cal.get("Earnings High")),
            "revenue_est": to_float(cal.get("Revenue Average")),
            "ex_dividend": str(cal.get("Ex-Dividend Date") or "") or None,
            "dividend_date": str(cal.get("Dividend Date") or "") or None,
        }

        # histórico de resultados (estimado vs reportado)
        edf = _safe(lambda: t.earnings_dates)
        if edf is not None and not edf.empty:
            hist = []
            for ts_, row in edf.head(10).iterrows():
                hist.append({
                    "date": ts_.strftime("%Y-%m-%d"),
                    "eps_est": to_float(row.get("EPS Estimate")),
                    "eps_real": to_float(row.get("Reported EPS")),
                    "surprise_pct": to_float(row.get("Surprise(%)")),
                })
            out["earnings_history"] = hist

        # participação acionária
        mh = _safe(lambda: t.major_holders)
        if mh is not None and not mh.empty:
            vals = mh["Value"].to_dict() if "Value" in mh else {}
            out["ownership"] = {
                "insiders_pct": to_float(vals.get("insidersPercentHeld")),
                "institutions_pct": to_float(vals.get("institutionsPercentHeld")),
                "institutions_count": to_float(vals.get("institutionsCount")),
            }
        ih = _safe(lambda: t.institutional_holders)
        if ih is not None and not ih.empty:
            out["top_holders"] = [{
                "holder": r.get("Holder"),
                "pct": to_float(r.get("pctHeld")),
                "shares": to_float(r.get("Shares")),
                "value": to_float(r.get("Value")),
                "date": str(r.get("Date Reported"))[:10],
            } for _, r in ih.head(10).iterrows()]

        # insider transactions
        ins = _safe(lambda: t.insider_transactions)
        if ins is not None and not ins.empty:
            out["insiders"] = [{
                "insider": r.get("Insider"),
                "position": r.get("Position"),
                "text": r.get("Text"),
                "shares": to_float(r.get("Shares")),
                "value": to_float(r.get("Value")),
                "date": str(r.get("Start Date"))[:10],
            } for _, r in ins.head(15).iterrows()]

        # dividendos: últimos pagamentos + soma anual
        div = _safe(lambda: t.dividends)
        if div is not None and len(div):
            out["dividends"] = {
                "recent": [{"date": ts_.strftime("%Y-%m-%d"), "amount": to_float(v)}
                           for ts_, v in div.tail(12).items()],
                "by_year": [{"year": int(y), "total": round(float(s), 4)}
                            for y, s in div.groupby(div.index.year).sum().tail(6).items()],
            }

        # pares do setor
        out["peers"] = _safe(lambda: _peers(yf_sym, info.get("sector"),
                                            out["mcap"]), [])
        out["smart_money"] = _safe(
            lambda: insider_data.smart_money(
                yf_sym, info.get("country"), info.get("quoteType")),
            {"feeds": [], "sources": [], "errors": ["fontes indisponíveis"]})
        return out

    return _cached(("ea", yf_sym), 900, fetch)
