"""
CDTY — painel de commodities: cotações por grupo, curvas futuras (contratos
individuais do Yahoo, ex.: CLZ26.NYM), spreads entre vencimentos, clima nas
regiões produtoras (met.no, sem chave), frete/shipping (proxies de mercado)
e estoques/produção via EIA (opcional, exige EIA_API_KEY grátis no .env —
FRED e Open-Meteo são bloqueados nesta rede).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

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


# ── grupos do painel ─────────────────────────────────────────────────────

GROUPS = [
    ("Energia", [("CL=F", "Petróleo WTI"), ("BZ=F", "Brent"),
                 ("NG=F", "Gás Natural"), ("RB=F", "Gasolina RBOB"),
                 ("HO=F", "Óleo p/ aquecimento")]),
    ("Metais", [("GC=F", "Ouro"), ("SI=F", "Prata"), ("PL=F", "Platina"),
                ("PA=F", "Paládio"), ("HG=F", "Cobre")]),
    ("Grãos", [("ZC=F", "Milho"), ("ZW=F", "Trigo"), ("ZS=F", "Soja"),
               ("ZM=F", "Farelo de soja"), ("ZL=F", "Óleo de soja")]),
    ("Softs", [("KC=F", "Café"), ("SB=F", "Açúcar"), ("CC=F", "Cacau"),
               ("CT=F", "Algodão"), ("LE=F", "Boi gordo"), ("OJ=F", "Suco de laranja")]),
]


def overview() -> dict:
    def fetch():
        from providers import tradfi_data
        all_syms = [s for _, items in GROUPS for s, _ in items]
        q = tradfi_data.quotes(all_syms)
        out = []
        for gname, items in GROUPS:
            rows = []
            for sym, label in items:
                r = q.get(sym)
                if r:
                    rows.append({**r, "base": sym, "label": label})
            out.append({"group": gname, "rows": rows})
        return {"groups": out, "ts": int(time.time() * 1000)}

    return _cached(("cdty_overview",), 60, fetch)


# ── curvas futuras ───────────────────────────────────────────────────────

_MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
                7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}

# raiz -> (exchange, rótulo, meses ativos, unidade)
CURVES = {
    "CL": ("NYM", "Petróleo WTI", list(range(1, 13)), "USD/bbl"),
    "BZ": ("NYM", "Brent", list(range(1, 13)), "USD/bbl"),
    "NG": ("NYM", "Gás Natural", list(range(1, 13)), "USD/MMBtu"),
    "RB": ("NYM", "Gasolina RBOB", list(range(1, 13)), "USD/gal"),
    "GC": ("CMX", "Ouro", [2, 4, 6, 8, 10, 12], "USD/oz"),
    "SI": ("CMX", "Prata", [3, 5, 7, 9, 12], "USD/oz"),
    "HG": ("CMX", "Cobre", [3, 5, 7, 9, 12], "USD/lb"),
    "ZC": ("CBT", "Milho", [3, 5, 7, 9, 12], "¢/bu"),
    "ZW": ("CBT", "Trigo", [3, 5, 7, 9, 12], "¢/bu"),
    "ZS": ("CBT", "Soja", [1, 3, 5, 7, 8, 9, 11], "¢/bu"),
    "KC": ("NYB", "Café", [3, 5, 7, 9, 12], "¢/lb"),
    "SB": ("NYB", "Açúcar", [3, 5, 7, 10], "¢/lb"),
    "CC": ("NYB", "Cacau", [3, 5, 7, 9, 12], "USD/t"),
    "CT": ("NYB", "Algodão", [3, 5, 7, 10, 12], "¢/lb"),
}


def _contract_symbols(root: str, n: int = 8) -> list:
    exch, _, months, _ = CURVES[root]
    today = date.today()
    out = []
    y, m = today.year, today.month + 1  # começa no próximo mês (front ~ =F)
    while len(out) < n and y < today.year + 4:
        if m > 12:
            m -= 12
            y += 1
        if m in months:
            code = _MONTH_CODES[m]
            out.append((f"{root}{code}{str(y)[2:]}.{exch}",
                        f"{code}{str(y)[2:]}", date(y, m, 1)))
        m += 1
    return out


def futures_curve(root: str) -> dict:
    root = root.upper()
    if root not in CURVES:
        raise ValueError(f"curva deve ser uma de {list(CURVES)}")

    def fetch():
        import yfinance as yf
        exch, label, _, unit = CURVES[root]
        contracts = _contract_symbols(root)

        def one(c):
            sym, code, dt = c
            try:
                px = _f(yf.Ticker(sym).fast_info["last_price"])
                return {"symbol": sym, "code": code, "month": dt.strftime("%Y-%m"),
                        "price": px}
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            pts = [r for r in pool.map(one, contracts) if r and r["price"]]

        spot = None
        try:
            spot = _f(yf.Ticker(f"{root}=F").fast_info["last_price"])
        except Exception:
            pass

        spreads = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            spreads.append({
                "pair": f"{a['code']}-{b['code']}",
                "value": round(a["price"] - b["price"], 4),
            })
        structure = None
        if len(pts) >= 2:
            structure = "contango" if pts[-1]["price"] > pts[0]["price"] else "backwardation"

        return {"root": root, "label": label, "unit": unit, "spot": spot,
                "points": pts, "spreads": spreads, "structure": structure,
                "ts": int(time.time() * 1000)}

    return _cached(("curve", root), 900, fetch)


def curves_meta() -> list:
    return [{"key": k, "label": v[1], "unit": v[3]} for k, v in CURVES.items()]


# ── clima nas regiões produtoras (met.no) ────────────────────────────────

REGIONS = [
    ("Corn Belt — Iowa, EUA", 41.9, -93.6, "milho · soja"),
    ("Mato Grosso, Brasil", -12.6, -55.7, "soja · milho · algodão"),
    ("Sul de Minas, Brasil", -21.5, -45.4, "café"),
    ("Pampas, Argentina", -34.6, -60.9, "soja · trigo · milho"),
    ("Costa do Marfim", 6.8, -5.3, "cacau"),
    ("Punjab, Índia", 30.9, 75.8, "trigo · arroz"),
    ("Ribeirão Preto, Brasil", -21.2, -47.8, "açúcar (cana)"),
    ("Kansas, EUA", 38.5, -98.0, "trigo"),
]


def _metno_region(name, lat, lon, crops):
    import requests
    r = requests.get(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params={"lat": lat, "lon": lon},
        headers={"User-Agent": "graph-terminal/1.0 contato@graph.local"},
        timeout=20)
    r.raise_for_status()
    series = r.json()["properties"]["timeseries"]
    horizon = datetime.now(timezone.utc) + timedelta(days=7)
    precip, tmax = 0.0, None
    for e in series:
        ts = datetime.fromisoformat(e["time"].replace("Z", "+00:00"))
        if ts > horizon:
            break
        d = e.get("data", {})
        nxt = d.get("next_1_hours") or d.get("next_6_hours")
        if nxt:
            precip += _f(nxt.get("details", {}).get("precipitation_amount")) or 0
        t = _f(d.get("instant", {}).get("details", {}).get("air_temperature"))
        if t is not None and (tmax is None or t > tmax):
            tmax = t
    flags = []
    if precip < 5:
        flags.append("seca")
    elif precip > 80:
        flags.append("chuva excessiva")
    if tmax is not None and tmax >= 35:
        flags.append("calor extremo")
    # explicação didática: condição climática → efeito na safra → efeito no preço
    impact_map = {
        "seca": ("sem chuva a planta sofre e a colheita encolhe — oferta menor "
                 "com a mesma demanda: preço tende a SUBIR"),
        "chuva excessiva": ("chuva demais atrasa plantio/colheita e apodrece o que "
                            "está no campo — oferta menor e de pior qualidade: "
                            "preço tende a SUBIR"),
        "calor extremo": ("calor ≥35°C estressa a planta e queima a floração (fase "
                          "que vira fruto/grão) — safra futura menor: preço tende a SUBIR"),
    }
    impact = ("; ".join(impact_map[f] for f in flags) if flags else
              "clima dentro do normal — safra sem estresse, sem pressão de preço vinda desta região")
    return {"region": name, "crops": crops, "precip_7d_mm": round(precip, 1),
            "tmax_7d": tmax, "flags": flags, "impact": impact}


def weather() -> dict:
    def fetch():
        rows, failed = [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_metno_region, *r): r[0] for r in REGIONS}
            for fut, name in futs.items():
                try:
                    rows.append(fut.result())
                except Exception:
                    failed.append(name)
        rows.sort(key=lambda r: r["precip_7d_mm"])
        return {"rows": rows, "failed": failed, "source": "met.no (previsão 7 dias)",
                "ts": int(time.time() * 1000)}

    return _cached(("weather",), 3600, fetch)


# ── frete & shipping (proxies de mercado) ────────────────────────────────

SHIPPING = [
    ("BDRY", "ETF de futuros de frete seco (Baltic)", "dry bulk"),
    ("GOGL", "Golden Ocean", "dry bulk"),
    ("SBLK", "Star Bulk", "dry bulk"),
    ("FRO", "Frontline", "petroleiros"),
    ("STNG", "Scorpio Tankers", "produtos"),
    ("TNK", "Teekay Tankers", "petroleiros"),
    ("ZIM", "ZIM Integrated", "contêineres"),
    ("MATX", "Matson", "contêineres"),
    ("DAC", "Danaos", "contêineres"),
]


def shipping() -> dict:
    def fetch():
        from providers import tradfi_data
        q = tradfi_data.quotes([s for s, *_ in SHIPPING])
        rows = []
        for sym, label, seg in SHIPPING:
            r = q.get(sym)
            if r:
                rows.append({**r, "base": sym, "label": label, "segment": seg})
        return {"rows": rows,
                "note": ("índices Baltic (BDI/BDTI) são pagos — proxies: ETF de "
                         "futuros de frete (BDRY) e ações de armadores por segmento"),
                "ts": int(time.time() * 1000)}

    return _cached(("shipping",), 120, fetch)


# ── estoques / produção / demanda (EIA — chave grátis opcional) ──────────

_EIA_SERIES = [
    ("petroleum/stoc/wstk", "WCESTUS1", "Estoques de petróleo bruto (EUA)", "mil bbl"),
    ("petroleum/sum/sndw", "WCRFPUS2", "Produção de petróleo (EUA)", "mil bbl/d"),
    ("petroleum/sum/sndw", "WRPUPUS2", "Demanda — produtos entregues (EUA)", "mil bbl/d"),
    ("natural-gas/stor/wkly", "NW2_EPG0_SWO_R48_BCF", "Estoques de gás natural (EUA)", "Bcf"),
]


def inventories() -> dict:
    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key:
        return {
            "configured": False,
            "howto": ("Estoques/produção/demanda usam a API da EIA (gratuita). "
                      "Crie a chave em https://www.eia.gov/opendata/register.php "
                      "e adicione EIA_API_KEY=... no .env"),
        }

    def fetch():
        import requests
        series = []
        for route, sid, label, unit in _EIA_SERIES:
            try:
                r = requests.get(
                    f"https://api.eia.gov/v2/{route}/data/",
                    params={"api_key": key, "frequency": "weekly",
                            "data[0]": "value", "facets[series][]": sid,
                            "sort[0][column]": "period",
                            "sort[0][direction]": "desc", "length": 60},
                    timeout=20)
                rows = r.json()["response"]["data"]
                rows.reverse()
                series.append({
                    "id": sid, "label": label, "unit": unit,
                    "dates": [x["period"] for x in rows],
                    "values": [_f(x["value"]) for x in rows],
                    "now": _f(rows[-1]["value"]) if rows else None,
                })
            except Exception:
                series.append({"id": sid, "label": label, "error": True})
        return {"configured": True, "series": series,
                "ts": int(time.time() * 1000)}

    return _cached(("eia",), 3600, fetch)
