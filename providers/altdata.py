"""
Dados alternativos e inteligência de mercado.

Fontes GRATUITAS validadas nesta rede (FRED e Open-Meteo são bloqueados):
- NY Fed GSCPI (Global Supply Chain Pressure Index) — série mensal REAL,
  z-score por construção (xls público do newyorkfed.org).
- TSA checkpoint travel numbers — passageiros/dia em aeroportos dos EUA
  (proxy de tráfego aéreo/demanda de viagem; tsa.gov, tabela HTML).
- NOAA CPC ONI — índice ENSO (El Niño/La Niña), impacto em safras.
- met.no — clima nas regiões produtoras (já usado em commodities_data).
- ccxt/Bybit — funding rates dos ~600 perps num shot + open interest dos
  majors (visão agregada difícil de achar pronta).
- yfinance — balanços trimestrais p/ métricas setoriais (estoques, dias de
  estoque, margem) e cestas de proxies p/ índices compostos.

Indicadores PROPRIETÁRIOS (metodologia declarada, z-scores sobre ~1 ano):
- Risk-on/off composto (cobre/ouro, HYG/TLT, VIX, DXY, momentum BTC)
- Estresse de supply chain (GSCPI + BDRY + armadores + tankers + WTI)
- Nowcast de inflação por commodities (momentum 63d ponderado por cesta)

Honestidade: tudo que é proxy/heurística é rotulado; USDA é bloqueado nesta
rede e dados pagos (AIS de navios, cartões de crédito, satélite) não entram.
"""

from __future__ import annotations

import io
import math
import re
import time

import pandas as pd

_UA = {"User-Agent": "GraphQuantLab/1.0 henrique.de.paula.valim@gmail.com"}
_cache: dict = {}


def _cached(key, ttl_s, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl_s:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _f(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ── séries auxiliares (yfinance em lote) ─────────────────────────────────

def _closes(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    def fetch():
        import yfinance as yf
        df = yf.download(tickers, period=period, interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        return df.dropna(how="all")

    return _cached(("closes", tuple(sorted(tickers)), period), 1800, fetch)


def _z(s: pd.Series, window: int = 126) -> pd.Series:
    """z-score da série vs a própria janela recente."""
    s = s.dropna()
    m = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - m) / sd


def _ser(s: pd.Series, n: int = 126) -> dict:
    s = s.dropna().tail(n)
    return {"ts": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": [round(float(v), 3) for v in s]}


# ── Indicadores proprietários ────────────────────────────────────────────

def indicators() -> dict:
    return _cached("indicators", 1800, _indicators)


def _indicators() -> dict:
    out = {"indicators": [], "note": ("índices PROPRIETÁRIOS calculados de "
           "proxies públicos — metodologia declarada em cada card")}

    # 1) Risk-on / Risk-off
    try:
        tk = ["HG=F", "GC=F", "HYG", "TLT", "^VIX", "DX-Y.NYB", "BTC-USD"]
        c = _closes(tk).ffill()
        comps = {
            "Cobre/Ouro": (_z(c["HG=F"] / c["GC=F"]), +1),
            "Crédito HY (HYG/TLT)": (_z(c["HYG"] / c["TLT"]), +1),
            "VIX (invertido)": (_z(c["^VIX"]), -1),
            "Dólar DXY (invertido)": (_z(c["DX-Y.NYB"]), -1),
            "Momentum BTC 30d": (_z(c["BTC-USD"].pct_change(30)), +1),
        }
        df = pd.DataFrame({k: z * sgn for k, (z, sgn) in comps.items()}).dropna()
        composite = df.mean(axis=1)
        out["indicators"].append({
            "id": "risk", "name": "Risk-on / Risk-off",
            "value": round(float(composite.iloc[-1]), 2),
            "unit": "z", "range": [-2.5, 2.5],
            "reading": ("RISK-ON" if composite.iloc[-1] > 0.5 else
                        "RISK-OFF" if composite.iloc[-1] < -0.5 else "NEUTRO"),
            "components": [{"name": k, "z": round(float(df[k].iloc[-1]), 2)}
                           for k in df.columns],
            "history": _ser(composite),
            "method": ("média dos z-scores (126d): razão cobre/ouro, razão "
                       "HYG/TLT, VIX e DXY invertidos, momentum 30d do BTC"),
        })
    except Exception as e:
        out["indicators"].append({"id": "risk", "name": "Risk-on / Risk-off",
                                  "error": str(e)[:200]})

    # 2) Estresse de supply chain (ancorado no GSCPI real)
    try:
        gs = gscpi_series()
        tk = ["BDRY", "ZIM", "HLAG.DE", "MAERSK-B.CO", "FRO", "STNG", "CL=F"]
        c = _closes(tk).ffill()
        container = c[["ZIM", "HLAG.DE", "MAERSK-B.CO"]].pct_change(63).mean(axis=1)
        tanker = c[["FRO", "STNG"]].pct_change(63).mean(axis=1)
        comps = {
            "GSCPI (NY Fed, real)": float(gs["values"][-1]),
            "Frete seco (BDRY, z)": round(float(_z(c["BDRY"]).iloc[-1]), 2),
            "Armadores contêiner (mom 63d, z)": round(float(_z(container).iloc[-1]), 2),
            "Tankers (mom 63d, z)": round(float(_z(tanker).iloc[-1]), 2),
            "Petróleo WTI (z)": round(float(_z(c["CL=F"]).iloc[-1]), 2),
        }
        vals = [v for v in comps.values() if v is not None]
        val = sum(vals) / len(vals)
        out["indicators"].append({
            "id": "supply", "name": "Estresse de Supply Chain",
            "value": round(val, 2), "unit": "z", "range": [-2.5, 2.5],
            "reading": ("PRESSÃO ALTA" if val > 0.5 else
                        "FOLGA" if val < -0.5 else "NORMAL"),
            "components": [{"name": k, "z": v} for k, v in comps.items()],
            "history": None,
            "method": ("média de: GSCPI oficial (já é z-score) + z de BDRY, "
                       "momentum de armadores/tankers e WTI (proxies)"),
        })
    except Exception as e:
        out["indicators"].append({"id": "supply", "name": "Estresse de Supply Chain",
                                  "error": str(e)[:200]})

    # 3) Nowcast de inflação por commodities
    try:
        basket = {"CL=F": 0.25, "RB=F": 0.15, "NG=F": 0.10,   # energia 50%
                  "ZW=F": 0.10, "ZC=F": 0.10, "ZS=F": 0.10,   # grãos 30%
                  "HG=F": 0.12, "GC=F": 0.08}                  # metais 20%
        c = _closes(list(basket)).ffill()
        mom = sum(c[t].pct_change(63) * w for t, w in basket.items())
        val = float(mom.iloc[-1]) * 100
        out["indicators"].append({
            "id": "inflation", "name": "Nowcast de Inflação (commodities)",
            "value": round(val, 2), "unit": "% em 3m", "range": [-15, 15],
            "reading": ("PRESSÃO INFLACIONÁRIA" if val > 4 else
                        "DESINFLAÇÃO" if val < -4 else "ESTÁVEL"),
            "components": [{"name": t, "z": round(float(c[t].pct_change(63).iloc[-1] * 100), 1)}
                           for t in basket],
            "history": _ser(mom * 100),
            "method": ("variação 63d (~3 meses) da cesta ponderada: energia "
                       "50% (WTI, gasolina, gás), grãos 30%, metais 20% — "
                       "antecipa o componente de bens do CPI, NÃO é o CPI"),
        })
    except Exception as e:
        out["indicators"].append({"id": "inflation",
                                  "name": "Nowcast de Inflação", "error": str(e)[:200]})
    return out


# ── Supply chain (GSCPI real + proxies de frete/logística) ───────────────

def gscpi_series() -> dict:
    def fetch():
        import requests
        b = requests.get(
            "https://www.newyorkfed.org/medialibrary/research/interactives/"
            "gscpi/downloads/gscpi_data.xlsx",
            headers=_UA, timeout=25).content
        df = pd.read_excel(io.BytesIO(b), sheet_name="GSCPI Monthly Data")
        df = df.dropna(subset=["GSCPI"])
        return {"ts": [pd.Timestamp(d).strftime("%Y-%m") for d in df["Date"]],
                "values": [round(float(v), 3) for v in df["GSCPI"]]}

    return _cached("gscpi", 12 * 3600, fetch)


_FREIGHT_PROXIES = [
    ("BDRY", "Frete seco (ETF Baltic Dry)"), ("ZIM", "ZIM (contêiner)"),
    ("HLAG.DE", "Hapag-Lloyd (contêiner)"), ("MAERSK-B.CO", "Maersk (contêiner)"),
    ("FRO", "Frontline (tanker cru)"), ("STNG", "Scorpio (tanker produto)"),
    ("FDX", "FedEx (logística)"), ("UPS", "UPS (logística)"),
    ("CHRW", "C.H. Robinson (freight broker)"), ("EXPD", "Expeditors (freight fwd)"),
]


def supply_chain() -> dict:
    def fetch():
        gs = gscpi_series()
        last = gs["values"][-1]
        yoy = last - gs["values"][-13] if len(gs["values"]) > 13 else None
        from providers import tradfi_data
        q = tradfi_data.quotes([t for t, _ in _FREIGHT_PROXIES])
        rows = [{"symbol": t, "label": lbl,
                 "last": (q.get(t) or {}).get("last"),
                 "pct24h": (q.get(t) or {}).get("pct24h")}
                for t, lbl in _FREIGHT_PROXIES]
        return {
            "gscpi": {**gs, "last": last, "yoy_delta": round(yoy, 2) if yoy is not None else None,
                      "source": "NY Fed (dado oficial, mensal, z-score)"},
            "proxies": rows,
            "note": ("proxies de mercado: ações de armadores/logística e ETF "
                     "de frete refletem expectativas, não o frete spot em si; "
                     "índices Baltic/Freightos spot são pagos"),
        }

    return _cached("supplychain", 1800, fetch)


# ── Tráfego (TSA — passageiros/dia nos EUA) ──────────────────────────────

def traffic() -> dict:
    def fetch():
        import requests
        html = requests.get("https://www.tsa.gov/travel/passenger-volumes",
                            headers=_UA, timeout=20).text
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            tds = re.findall(r"<td[^>]*>\s*([^<]*?)\s*</td>", tr)
            if len(tds) >= 2 and re.match(r"\d+/\d+/\d+", tds[0]):
                n = _f(tds[1].replace(",", ""))
                if n:
                    rows.append((pd.Timestamp(tds[0]).strftime("%Y-%m-%d"), n))
        rows.sort()
        s = pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()
        avg7 = s.rolling(7).mean()
        # variação: média 7d atual vs média 7d de ~1 mês atrás
        mom = (avg7.iloc[-1] / avg7.iloc[-29] - 1) * 100 if len(avg7) > 29 else None
        return {
            "ts": [d.strftime("%Y-%m-%d") for d in s.index],
            "passengers": [int(v) for v in s],
            "avg7": [round(float(v)) if not math.isnan(v) else None for v in avg7],
            "last": int(s.iloc[-1]), "last_date": s.index[-1].strftime("%Y-%m-%d"),
            "avg7_last": round(float(avg7.iloc[-1])),
            "mom_pct": round(mom, 1) if mom is not None else None,
            "source": ("TSA checkpoint travel numbers (oficial, diário) — "
                       "proxy de demanda de viagem aérea nos EUA"),
        }

    return _cached("tsa", 6 * 3600, fetch)


# ── Clima (ENSO oficial + regiões produtoras via met.no) ────────────────

def climate() -> dict:
    def fetch():
        import requests
        txt = requests.get(
            "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
            headers=_UA, timeout=20).text
        rows = []
        for ln in txt.strip().splitlines()[1:]:
            parts = ln.split()
            if len(parts) == 4:
                rows.append({"season": parts[0], "year": int(parts[1]),
                             "anom": _f(parts[3])})
        recent = rows[-12:]
        last = recent[-1]
        anom = last["anom"] or 0
        # convenção NOAA: 5 trimestres consecutivos ±0.5 = evento; aqui
        # classificamos a leitura corrente (rótulo aproximado)
        status = ("El Niño" if anom >= 0.5 else
                  "La Niña" if anom <= -0.5 else "Neutro")
        impact = {
            "El Niño": ("chuva no Sul do Brasil/Argentina, seca no Sudeste "
                        "Asiático/Austrália — pressão em açúcar, óleo de "
                        "palma, trigo australiano"),
            "La Niña": ("seca no Sul do Brasil/Argentina (soja/milho), chuva "
                        "na Austrália/Indonésia — pressão em grãos americanos"),
            "Neutro": "sem sinal ENSO dominante nas safras",
        }[status]
        weather = None
        try:
            from providers import commodities_data
            weather = commodities_data.weather()
        except Exception:
            pass
        return {
            "enso": {"status": status, "anom": anom,
                     "season": f"{last['season']} {last['year']}",
                     "recent": recent, "impact": impact,
                     "source": "NOAA CPC ONI (oficial, trimestral móvel)"},
            "regions": weather,
        }

    return _cached("climate", 12 * 3600, fetch)


# ── Métricas setoriais (estoques/margens dos balanços trimestrais) ──────

SECTORS = {
    "varejo": {"label": "Varejo (EUA)",
               "tickers": ["WMT", "TGT", "COST", "HD", "AMZN"],
               "insight": ("dias de estoque subindo = demanda fraca ou "
                           "excesso de compra → risco de desconto/margem")},
    "semis": {"label": "Semicondutores",
              "tickers": ["NVDA", "TSM", "INTC", "MU", "AMD"],
              "insight": ("estoque de chips é o ciclo do setor: dias subindo "
                          "= glut (2022-23); caindo = escassez/pricing power")},
    "autos": {"label": "Automóveis",
              "tickers": ["TSLA", "F", "GM", "TM"],
              "insight": "estoque alto em montadora = corte de produção à frente"},
    "energia": {"label": "Energia (integradas)",
                "tickers": ["XOM", "CVX", "COP", "SLB"],
                "insight": ("margem bruta acompanha o crack spread; estoques "
                            "físicos de petróleo estão na aba EIA do CDTY")},
}


def _sector_row(t: str) -> dict | None:
    import yfinance as yf
    tk = yf.Ticker(t)
    bs = tk.quarterly_balance_sheet
    inc = tk.quarterly_income_stmt
    if bs is None or bs.empty or inc is None or inc.empty:
        return None

    def row(df, name):
        return df.loc[name].dropna() if name in df.index else pd.Series(dtype=float)

    inv = row(bs, "Inventory")
    cogs = row(inc, "Cost Of Revenue")
    rev = row(inc, "Total Revenue")
    gp = row(inc, "Gross Profit")
    quarters = sorted(set(inv.index) & set(cogs.index), reverse=True)[:5]
    days_hist = []
    for q in quarters:
        if cogs.get(q):
            days_hist.append({"q": pd.Timestamp(q).strftime("%Y-%m"),
                              "days": round(float(inv[q] / cogs[q] * 91.25), 1)})
    rev_yoy = None
    if len(rev) >= 5 and rev.iloc[4]:
        rev_yoy = (float(rev.iloc[0]) / float(rev.iloc[4]) - 1) * 100
    margin = (float(gp.iloc[0]) / float(rev.iloc[0]) * 100
              if len(gp) and len(rev) and rev.iloc[0] else None)
    d_now = days_hist[0]["days"] if days_hist else None
    d_yoy = days_hist[4]["days"] if len(days_hist) > 4 else None
    # Yahoo às vezes mistura receita ACUMULADA do ano fiscal com COGS
    # trimestral (visto no MU) — sinaliza valores implausíveis em vez de
    # apresentá-los como fato
    suspect = ((margin is not None and margin > 85)
               or (rev_yoy is not None and abs(rev_yoy) > 200))
    return {
        "symbol": t,
        "inventory": float(inv.iloc[0]) if len(inv) else None,
        "days_inventory": d_now,
        "days_delta_yoy": (round(d_now - d_yoy, 1)
                           if d_now is not None and d_yoy is not None else None),
        "days_hist": list(reversed(days_hist)),
        "gross_margin_pct": round(margin, 1) if margin is not None else None,
        "rev_yoy_pct": round(rev_yoy, 1) if rev_yoy is not None else None,
        "suspect": suspect,
    }


def sector_metrics(sector: str) -> dict:
    if sector not in SECTORS:
        raise ValueError(f"setor deve ser um de {list(SECTORS)}")

    def fetch():
        from concurrent.futures import ThreadPoolExecutor
        cfg = SECTORS[sector]
        with ThreadPoolExecutor(max_workers=5) as pool:
            rows = list(pool.map(lambda t: _sector_row(t), cfg["tickers"]))
        rows = [r for r in rows if r]
        return {"sector": sector, "label": cfg["label"],
                "insight": cfg["insight"], "rows": rows,
                "sectors": {k: v["label"] for k, v in SECTORS.items()},
                "source": ("balanços/DRE trimestrais via Yahoo (filings "
                           "SEC/IFRS) — dado duro, ~45d de defasagem do "
                           "fechamento do trimestre")}

    return _cached(("sector", sector), 6 * 3600, fetch)


# ── Microestrutura cripto (funding agregado + open interest) ────────────

_MAJORS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK",
           "AVAX", "SUI", "LTC", "DOT"]


def crypto_micro() -> dict:
    def fetch():
        from concurrent.futures import ThreadPoolExecutor
        from providers.market_data import get_exchange
        ex = get_exchange("bybit")
        frs = ex.fetch_funding_rates()
        rows = [(s.split("/")[0], _f(v.get("fundingRate")))
                for s, v in frs.items()
                if s.endswith(":USDT") and _f(v.get("fundingRate")) is not None]
        rates = [r for _, r in rows]
        pos = sum(1 for r in rates if r > 0)
        rows.sort(key=lambda x: x[1])

        def oi_one(base):
            try:
                t = ex.fetch_ticker(f"{base}/USDT:USDT")
                oi = ex.fetch_open_interest(f"{base}/USDT:USDT")
                amt = _f(oi.get("openInterestAmount"))
                last = _f(t.get("last"))
                fr = _f((frs.get(f"{base}/USDT:USDT") or {}).get("fundingRate"))
                return {"symbol": base, "oi": amt, "last": last,
                        "oi_usd": amt * last if amt and last else None,
                        "funding_pct": fr * 100 if fr is not None else None,
                        "pct24h": _f(t.get("percentage"))}
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            majors = [r for r in pool.map(oi_one, _MAJORS) if r]
        return {
            "n_perps": len(rates),
            "pct_positive": round(pos / len(rates) * 100, 1) if rates else None,
            "mean_funding_pct": (round(sum(rates) / len(rates) * 100, 4)
                                 if rates else None),
            "extremes": {
                "negative": [{"symbol": s, "funding_pct": round(r * 100, 4)}
                             for s, r in rows[:5]],
                "positive": [{"symbol": s, "funding_pct": round(r * 100, 4)}
                             for s, r in rows[-5:][::-1]],
            },
            "majors": sorted(majors, key=lambda m: -(m["oi_usd"] or 0)),
            "note": ("funding de TODOS os perps USDT da Bybit num snapshot — "
                     "% positivo alto = mercado comprado/alavancado (contrar."
                     " baixista); extremos negativos = shorts pagando p/ "
                     "manter posição (squeeze possível)"),
            "ts": int(time.time() * 1000),
        }

    return _cached("cryptomicro", 300, fetch)
