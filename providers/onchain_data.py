"""
Dados ON-CHAIN de cripto — fontes gratuitas validadas nesta rede:

- blockchain.info charts  : séries 180d da rede Bitcoin (hashrate, endereços
                            ativos, nº de transações) — sem chave.
- mempool.space           : fees recomendadas (sat/vB) e ajuste de
                            dificuldade em andamento — sem chave.
- Blockchair              : snapshot da rede BTC (mempool, circulação,
                            dominância, fee média) — free tier sem chave.
- DeFiLlama               : TVL por chain + stablecoins (circulante e Δ30d
                            = liquidez entrando/saindo do ecossistema).
- alternative.me          : Fear & Greed index (série 90d).
- CoinGecko (free)        : mcap global, dominâncias, variação 24h.

Honestidade: métricas pagas (SOPR, MVRV, reservas em exchanges da
Glassnode/CryptoQuant) NÃO entram — só o que é público e verificável.
Tudo agregado em um payload único com cache de 15 min.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from common import ttl_cache, to_float

_cached = ttl_cache()
_UA = {"User-Agent": "GraphQuantLab/1.0 henrique.de.paula.valim@gmail.com"}


def _get(url, timeout=15):
    import requests
    r = requests.get(url, headers=_UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


# visão POR MOEDA
# símbolo → id do CoinGecko (majors; fora do mapa cai no /search)
_CG_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "DOGE": "dogecoin", "BNB": "binancecoin", "ADA": "cardano",
    "LINK": "chainlink", "AVAX": "avalanche-2", "SUI": "sui",
    "LTC": "litecoin", "DOT": "polkadot", "TON": "the-open-network",
    "TRX": "tron", "NEAR": "near", "POL": "polygon-ecosystem-token",
    "MATIC": "polygon-ecosystem-token", "OP": "optimism", "ARB": "arbitrum",
    "PEPE": "pepe", "SHIB": "shiba-inu", "UNI": "uniswap", "AAVE": "aave",
    "ATOM": "cosmos", "XLM": "stellar", "HBAR": "hedera-hashgraph",
    "FIL": "filecoin", "APT": "aptos", "INJ": "injective-protocol",
}

# símbolo → chain do Blockchair (stats de rede por moeda)
_BLOCKCHAIR = {
    "BTC": "bitcoin", "ETH": "ethereum", "LTC": "litecoin",
    "DOGE": "dogecoin", "BCH": "bitcoin-cash", "XRP": "ripple",
    "XLM": "stellar", "ADA": "cardano", "XMR": "monero",
    "DASH": "dash", "ZEC": "zcash",
}

# símbolo → chain da DeFiLlama (série de TVL do ecossistema)
_LLAMA_CHAIN = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "BSC",
    "AVAX": "Avalanche", "TRX": "Tron", "ADA": "Cardano", "NEAR": "Near",
    "APT": "Aptos", "SUI": "Sui", "TON": "TON", "ARB": "Arbitrum",
    "OP": "OP Mainnet", "POL": "Polygon", "MATIC": "Polygon",
    "DOT": "Polkadot", "ATOM": "CosmosHub",
}


def coin(symbol: str) -> dict:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol obrigatório")
    return _cached(("coin", sym), 900, lambda: _coin(sym))


def _coin(sym: str) -> dict:
    out = {"symbol": sym, "errors": [], "ts": int(time.time() * 1000)}

    def safe(name, fn):
        try:
            return fn()
        except Exception as e:
            out["errors"].append(f"{name}: {str(e)[:100]}")
            return None

    cg_id = _CG_IDS.get(sym)
    if not cg_id:
        res = safe("search", lambda: _get(
            f"https://api.coingecko.com/api/v3/search?query={sym}"))
        coins = (res or {}).get("coins") or []
        hit = next((c for c in coins if c.get("symbol", "").upper() == sym),
                   coins[0] if coins else None)
        if not hit:
            raise ValueError(f"moeda {sym} não encontrada no CoinGecko")
        cg_id = hit["id"]

    def fetch_cg():
        return _get(f"https://api.coingecko.com/api/v3/coins/{cg_id}"
                    "?localization=false&tickers=false&market_data=true"
                    "&community_data=true&developer_data=true&sparkline=false",
                    timeout=20)

    def fetch_deriv():
        from providers.market_data import get_exchange
        ex = get_exchange("bybit")
        pair = f"{sym}/USDT:USDT"
        fr = ex.fetch_funding_rate(pair)
        oi = ex.fetch_open_interest(pair)
        t = ex.fetch_ticker(pair)
        amt, last = to_float(oi.get("openInterestAmount")), to_float(t.get("last"))
        return {"funding_pct": (to_float(fr.get("fundingRate")) or 0) * 100,
                "oi_usd": amt * last if amt and last else None,
                "last": last}

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_cg = pool.submit(safe, "coingecko", fetch_cg)
        f_chair = (pool.submit(safe, "blockchair", lambda: _get(
            f"https://api.blockchair.com/{_BLOCKCHAIR[sym]}/stats", timeout=20))
            if sym in _BLOCKCHAIR else None)
        f_tvl = (pool.submit(safe, "tvl", lambda: _get(
            "https://api.llama.fi/v2/historicalChainTvl/"
            + _LLAMA_CHAIN[sym].replace(" ", "%20"), timeout=20))
            if sym in _LLAMA_CHAIN else None)
        f_deriv = pool.submit(safe, "bybit", fetch_deriv)
        cg = f_cg.result()
        chair = f_chair.result() if f_chair else None
        tvl = f_tvl.result() if f_tvl else None
        deriv = f_deriv.result()

    if cg:
        md = cg.get("market_data") or {}
        circ, mx = to_float(md.get("circulating_supply")), to_float(md.get("max_supply"))
        dev = cg.get("developer_data") or {}
        com = cg.get("community_data") or {}
        out["profile"] = {
            "name": cg.get("name"), "rank": cg.get("market_cap_rank"),
            "price": to_float((md.get("current_price") or {}).get("usd")),
            "mcap": to_float((md.get("market_cap") or {}).get("usd")),
            "volume_24h": to_float((md.get("total_volume") or {}).get("usd")),
            "chg_7d_pct": to_float(md.get("price_change_percentage_7d")),
            "chg_30d_pct": to_float(md.get("price_change_percentage_30d")),
            "supply": {"circulating": circ, "max": mx,
                       "pct_emitted": circ / mx * 100 if circ and mx else None},
            "ath": {"price": to_float((md.get("ath") or {}).get("usd")),
                    "change_pct": to_float((md.get("ath_change_percentage") or {}).get("usd")),
                    "date": ((md.get("ath_date") or {}).get("usd") or "")[:10]},
            "sentiment_up_pct": to_float(cg.get("sentiment_votes_up_percentage")),
            "dev": {"stars": dev.get("stars"),
                    "commits_4w": dev.get("commit_count_4_weeks"),
                    "forks": dev.get("forks")},
            "community": {"twitter": com.get("twitter_followers"),
                          "reddit": com.get("reddit_subscribers")},
            "source": "CoinGecko",
        }
    if chair and chair.get("data"):
        d = chair["data"]
        out["network"] = {
            "txs_24h": d.get("transactions_24h"),
            "avg_fee_usd_24h": to_float(d.get("average_transaction_fee_usd_24h")),
            "mempool_txs": d.get("mempool_transactions"),
            "hashrate_24h": d.get("hashrate_24h"),
            "hodling_addresses": d.get("hodling_addresses"),
            "source": "Blockchair",
        }
    if tvl and isinstance(tvl, list):
        pts = tvl[-365:]
        out["tvl"] = {
            "chain": _LLAMA_CHAIN.get(sym),
            "ts": [p["date"] * 1000 for p in pts],
            "values": [to_float(p.get("tvl")) for p in pts],
            "last": to_float(pts[-1].get("tvl")) if pts else None,
            "chg_30d_pct": ((pts[-1]["tvl"] / pts[-31]["tvl"] - 1) * 100
                            if len(pts) > 31 and pts[-31].get("tvl") else None),
            "source": "DeFiLlama",
        }
    out["deriv"] = deriv
    return out


def _btc_metrics() -> dict:
    from providers import btc_onchain_metrics
    return btc_onchain_metrics.payload()


def _series_btc(chart: str, timespan: str = "180days") -> dict:
    d = _get(f"https://api.blockchain.info/charts/{chart}"
             f"?timespan={timespan}&format=json", timeout=20)
    vals = d.get("values") or []
    return {"ts": [v["x"] * 1000 for v in vals],
            "values": [to_float(v["y"]) for v in vals],
            "unit": d.get("unit")}


# payload único
def overview() -> dict:
    return _cached("onchain", 900, _overview)


def _overview() -> dict:
    out = {"btc": {}, "defi": {}, "sentiment": {}, "errors": [],
           "ts": int(time.time() * 1000)}

    def safe(name, fn):
        try:
            return fn()
        except Exception as e:
            out["errors"].append(f"{name}: {str(e)[:120]}")
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        f_hash = pool.submit(safe, "hashrate",
                             lambda: _series_btc("hash-rate"))
        f_addr = pool.submit(safe, "enderecos",
                             lambda: _series_btc("n-unique-addresses"))
        f_txs = pool.submit(safe, "transacoes",
                            lambda: _series_btc("n-transactions"))
        f_fees = pool.submit(safe, "fees", lambda: _get(
            "https://mempool.space/api/v1/fees/recommended"))
        f_diff = pool.submit(safe, "dificuldade", lambda: _get(
            "https://mempool.space/api/v1/difficulty-adjustment"))
        f_chair = pool.submit(safe, "blockchair", lambda: _get(
            "https://api.blockchair.com/bitcoin/stats", timeout=20))
        f_chains = pool.submit(safe, "defillama", lambda: _get(
            "https://api.llama.fi/v2/chains", timeout=20))
        f_stab = pool.submit(safe, "stablecoins", lambda: _get(
            "https://stablecoins.llama.fi/stablecoins?includePrices=false",
            timeout=25))
        f_fng = pool.submit(safe, "fear&greed", lambda: _get(
            "https://api.alternative.me/fng/?limit=90"))
        f_cg = pool.submit(safe, "coingecko", lambda: _get(
            "https://api.coingecko.com/api/v3/global", timeout=20))
        f_advanced = pool.submit(safe, "btc_metrics", _btc_metrics)

        hashrate, addresses, txs = f_hash.result(), f_addr.result(), f_txs.result()
        fees, diff = f_fees.result(), f_diff.result()
        chair = f_chair.result()
        chains, stables = f_chains.result(), f_stab.result()
        fng, cg = f_fng.result(), f_cg.result()
        advanced = f_advanced.result()

    # BTC: rede
    btc = {"hashrate": hashrate, "addresses": addresses, "txs": txs,
           "fees_satvb": fees, "source_series": "blockchain.info (diário, 180d)"}
    if diff:
        btc["difficulty"] = {
            "progress_pct": to_float(diff.get("progressPercent")),
            "change_pct": to_float(diff.get("difficultyChange")),
            "remaining_blocks": diff.get("remainingBlocks"),
            "retarget_ts": diff.get("estimatedRetargetDate"),
        }
    if chair and chair.get("data"):
        d = chair["data"]
        btc["snapshot"] = {
            "blocks": d.get("blocks"),
            "mempool_txs": d.get("mempool_transactions"),
            "circulation_btc": (to_float(d.get("circulation")) or 0) / 1e8 or None,
            "dominance_pct": to_float(d.get("market_dominance_percentage")),
            "avg_fee_usd_24h": to_float(d.get("average_transaction_fee_usd_24h")),
        }
    out["btc"] = btc

    # DeFi: TVL + stablecoins
    defi = {}
    if chains:
        rows = sorted(chains, key=lambda c: -(to_float(c.get("tvl")) or 0))
        defi["tvl_total"] = sum(to_float(c.get("tvl")) or 0 for c in chains)
        defi["chains"] = [{"name": c.get("name"), "tvl": to_float(c.get("tvl")),
                           "symbol": c.get("tokenSymbol")}
                          for c in rows[:12]]
    if stables and stables.get("peggedAssets"):
        assets = stables["peggedAssets"]

        def circ(a, key="circulating"):
            return to_float((a.get(key) or {}).get("peggedUSD")) or 0.0

        total_now = sum(circ(a) for a in assets)
        total_m = sum(circ(a, "circulatingPrevMonth") for a in assets)
        top = sorted(assets, key=lambda a: -circ(a))[:8]
        defi["stablecoins"] = {
            "total": total_now,
            "delta_30d": total_now - total_m if total_m else None,
            "delta_30d_pct": ((total_now / total_m - 1) * 100
                              if total_m else None),
            "top": [{
                "symbol": a.get("symbol"), "name": a.get("name"),
                "circulating": circ(a),
                "delta_30d_pct": ((circ(a) / circ(a, "circulatingPrevMonth") - 1) * 100
                                  if circ(a, "circulatingPrevMonth") else None),
            } for a in top],
            "note": ("supply de stablecoin crescendo = dinheiro novo entrando "
                     "no ecossistema (poder de compra em espera); encolhendo "
                     "= saída de liquidez"),
        }
    out["defi"] = defi

    # Sentimento / global
    sent = {}
    if fng and fng.get("data"):
        rows = fng["data"]                       # mais recente primeiro
        cur = rows[0]
        sent["fear_greed"] = {
            "value": int(cur["value"]),
            "label": cur.get("value_classification"),
            "series": {
                "ts": [int(r["timestamp"]) * 1000 for r in reversed(rows)],
                "values": [int(r["value"]) for r in reversed(rows)],
            },
        }
    if cg and cg.get("data"):
        d = cg["data"]
        mc = (d.get("total_market_cap") or {}).get("usd")
        vol = (d.get("total_volume") or {}).get("usd")
        dom = d.get("market_cap_percentage") or {}
        sent["global"] = {
            "mcap_usd": to_float(mc), "volume_24h_usd": to_float(vol),
            "mcap_change_24h_pct": to_float(d.get("market_cap_change_percentage_24h_usd")),
            "btc_dominance": to_float(dom.get("btc")),
            "eth_dominance": to_float(dom.get("eth")),
            "active_cryptos": d.get("active_cryptocurrencies"),
        }
    out["sentiment"] = sent
    out["btc_metrics"] = advanced or {
        "series": {}, "unavailable": [], "error": "métricas BTC indisponíveis"}
    return out
