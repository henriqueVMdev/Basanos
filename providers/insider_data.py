"""Regulatory insider filings and smart-money positioning by asset class."""
from __future__ import annotations

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor

import requests

_CACHE = {}
_UA = {"User-Agent": os.getenv("SEC_API_USER_AGENT", "GraphQuantLab contact@example.com")}

_CFTC = {
    # Yahoo symbol -> (display name, exact CFTC contract market code)
    "GC=F": ("GOLD", "088691"), "SI=F": ("SILVER", "084691"),
    "CL=F": ("CRUDE OIL", "067651"), "NG=F": ("NATURAL GAS", "023651"),
    "HG=F": ("COPPER", "085692"), "ZC=F": ("CORN", "002602"),
    "ZW=F": ("WHEAT-SRW", "001602"), "ZS=F": ("SOYBEANS", "005602"),
    "KC=F": ("COFFEE C", "083731"), "SB=F": ("SUGAR NO. 11", "080732"),
    "CC=F": ("COCOA", "073732"), "CT=F": ("COTTON NO. 2", "033661"),
}

_FAMOUS_BTC_WALLETS = [
    {"label": "Genesis block", "entity": "Satoshi Nakamoto (histórico)",
     "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "confidence": "verified",
     "note": "Destino convencionalmente exibido para a recompensa do bloco gênese."},
    {"label": "Bitcoin Pizza Day", "entity": "Compra das pizzas (histórico)",
     "address": "17SkEw2md5avVNyYgj6RiXuQKNwkXaxFyQ", "confidence": "verified",
     "note": "Endereço que recebeu os 10.000 BTC da transação de maio de 2010."},
    {"label": "Binance cold wallet", "entity": "Binance",
     "address": "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo", "confidence": "community_label",
     "note": "Rótulo público amplamente usado; custódia e finalidade podem mudar."},
]


def _get(url, **kwargs):
    r = requests.get(url, headers=_UA, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def _sec(ticker):
    tickers = _get("https://www.sec.gov/files/company_tickers.json")
    hit = next((x for x in tickers.values() if x["ticker"].upper() == ticker.upper()), None)
    if not hit:
        return None
    cik = str(hit["cik_str"]).zfill(10)
    data = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data.get("filings", {}).get("recent", {})
    filings = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in ("3", "4", "5"):
            continue
        acc = recent["accessionNumber"][i]
        primary = recent["primaryDocument"][i]
        filings.append({"form": form, "date": recent["filingDate"][i],
                        "description": recent.get("primaryDocDescription", [""] * (i + 1))[i],
                        "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{primary}"})
        if len(filings) >= 15:
            break
    return {"source": "SEC EDGAR", "kind": "regulatory", "filings": filings,
            "url": f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=only"}


def _cftc(symbol):
    contract = _CFTC.get(symbol.upper())
    if not contract:
        return None
    commodity, market_code = contract
    params = {"$limit": 104, "$order": "report_date_as_yyyy_mm_dd DESC",
              "$where": f"cftc_contract_market_code='{market_code}'"}
    rows = _get("https://publicreporting.cftc.gov/resource/6dca-aqww.json", params=params)
    points = []
    for r in reversed(rows):
        try:
            long_, short = float(r["noncomm_positions_long_all"]), float(r["noncomm_positions_short_all"])
            points.append({"date": r["report_date_as_yyyy_mm_dd"][:10], "net": long_ - short,
                           "long": long_, "short": short})
        except (KeyError, ValueError):
            continue
    if not points:
        return None
    nets = sorted(x["net"] for x in points)
    latest = points[-1]
    latest["net_percentile_2y"] = round(sum(x <= latest["net"] for x in nets) / len(nets) * 100, 1)
    return {"source": "CFTC Commitments of Traders", "kind": "positioning",
            "commodity": commodity, "contract_market_code": market_code,
            "latest": latest, "history": points,
            "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"}


def _crypto(symbol):
    base = symbol.upper().replace("-USD", "").replace("USDT", "")
    pair = base + "USDT"
    if base not in {"BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"}:
        return None
    params = {"symbol": pair, "period": "1d", "limit": 30}
    ratios = _get("https://fapi.binance.com/futures/data/topLongShortPositionRatio", params=params)
    points = [{"ts": int(x["timestamp"]), "ratio": float(x["longShortRatio"]),
               "long_pct": float(x.get("longAccount", x.get("longPosition", 0))) * 100,
               "short_pct": float(x.get("shortAccount", x.get("shortPosition", 0))) * 100}
              for x in ratios]
    result = {"source": "Binance Top Trader Long/Short", "kind": "smart_money_proxy",
            "note": "Top traders da exchange; não representa negociação de insider legal.",
            "latest": points[-1] if points else None, "history": points,
            "url": "https://www.binance.com/en/futures/"}
    if base == "BTC":
        result["famous_wallets"] = _btc_wallets()
    return result


def _wallet(address):
    stats = _get(f"https://mempool.space/api/address/{address}")
    txs = _get(f"https://mempool.space/api/address/{address}/txs")
    chain, mem = stats.get("chain_stats", {}), stats.get("mempool_stats", {})
    funded = int(chain.get("funded_txo_sum", 0)) + int(mem.get("funded_txo_sum", 0))
    spent = int(chain.get("spent_txo_sum", 0)) + int(mem.get("spent_txo_sum", 0))
    latest = txs[0] if txs else {}
    status = latest.get("status") or {}
    return {"balance_btc": (funded - spent) / 1e8, "received_btc": funded / 1e8,
            "spent_btc": spent / 1e8,
            "tx_count": int(chain.get("tx_count", 0)) + int(mem.get("tx_count", 0)),
            "mempool_tx_count": int(mem.get("tx_count", 0)),
            "last_txid": latest.get("txid"),
            "last_activity_ts": int(status["block_time"]) * 1000 if status.get("block_time") else None}


def _btc_wallets():
    wallets = list(_FAMOUS_BTC_WALLETS)
    # Optional user-maintained labels: JSON array with label/entity/address/note.
    try:
        custom = json.loads(os.getenv("BTC_FAMOUS_WALLETS_JSON", "[]"))
        wallets.extend(x for x in custom if isinstance(x, dict) and x.get("address"))
    except (TypeError, ValueError):
        pass
    def enrich(item):
        try:
            return {**item, **_wallet(item["address"]),
                    "explorer_url": f"https://mempool.space/address/{item['address']}"}
        except Exception as exc:
            return {**item, "error": str(exc)[:100],
                    "explorer_url": f"https://mempool.space/address/{item['address']}"}
    with ThreadPoolExecutor(max_workers=min(6, len(wallets))) as pool:
        return list(pool.map(enrich, wallets))


def _catalog(symbol, country):
    rows = [
        {"market": "Estados Unidos", "source": "SEC EDGAR Forms 3/4/5", "kind": "regulatory", "url": "https://www.sec.gov/edgar/search/"},
        {"market": "Brasil", "source": "CVM — Valores Mobiliários Negociados e Detidos", "kind": "regulatory", "url": "https://dados.cvm.gov.br/dataset/cia_aberta-doc-vlmo"},
        {"market": "Canadá", "source": "SEDI insider reports", "kind": "regulatory", "url": "https://www.sedi.ca/"},
        {"market": "Reino Unido", "source": "FCA / PDMR disclosures", "kind": "regulatory", "url": "https://data.fca.org.uk/"},
        {"market": "União Europeia", "source": "Regulador local / MAR PDMR", "kind": "regulatory", "url": "https://www.esma.europa.eu/"},
        {"market": "Commodities", "source": "CFTC Commitments of Traders", "kind": "positioning", "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"},
        {"market": "Cripto", "source": "Binance Top Traders + métricas on-chain", "kind": "smart_money_proxy", "url": "https://developers.binance.com/"},
    ]
    return rows


def smart_money(symbol: str, country=None, quote_type=None):
    key = (symbol.upper(), country, quote_type)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < 900:
        return hit[1]
    sym = symbol.upper()
    out = {"asset": sym, "feeds": [], "sources": _catalog(sym, country), "errors": []}
    jobs = []
    if sym in _CFTC:
        jobs.append(("CFTC", lambda: _cftc(sym)))
    elif sym.endswith("-USD") or sym in {"BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"}:
        jobs.append(("Binance", lambda: _crypto(sym)))
    elif "." not in sym and "=" not in sym:
        jobs.append(("SEC", lambda: _sec(sym)))
    for name, fn in jobs:
        try:
            value = fn()
            if value:
                out["feeds"].append(value)
        except Exception as exc:
            out["errors"].append(f"{name}: {str(exc)[:120]}")
    _CACHE[key] = (time.time(), out)
    return out
