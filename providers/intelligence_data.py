"""Decision layer: cruza TODAS as fontes que o app coleta para um ativo.

Fatores votam -1/0/+1 (BTC on-chain vota ±2 por agregar 7+ métricas com
backtest causal). O rótulo vem da soma; a confiança é a concordância entre os
fatores; a cobertura é a fração de fontes que respondeu. Divergências nascem
do cruzamento entre fatores; a validação usa retornos futuros reais do
próprio ativo ou o backtest do research — nunca opinião.

Fontes cruzadas por classe:
- todos: técnica (yfinance), sazonalidade, liquidez Fed (FRED), regime de
  risco (HY spread + VIX)
- cripto: on-chain BTC (research), funding perp Bybit, TVL DeFiLlama,
  Fear & Greed, amplitude de funding (todos os perps), top traders Binance,
  atividade dev CoinGecko
- ações EUA: filings de insiders SEC; aéreas: tráfego TSA; frete/armadores:
  GSCPI NY Fed
- commodities: posicionamento CFTC (COT); agrícolas: ENSO + clima nas
  regiões produtoras
"""
from __future__ import annotations
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from common import ttl_cache

_cached = ttl_cache()
_TRACK_DB = str(Path(__file__).parents[1] / "data" / "intelligence_signals.db")

# universo do ranking: cada classe demonstra os fatores que só ela tem
UNIVERSE = [
    ("BTC", "cripto"), ("ETH", "cripto"), ("SOL", "cripto"),
    ("XRP", "cripto"), ("BNB", "cripto"), ("DOGE", "cripto"),
    ("GC=F", "commodity"), ("SI=F", "commodity"), ("CL=F", "commodity"),
    ("NG=F", "commodity"), ("HG=F", "commodity"), ("ZC=F", "commodity"),
    ("ZW=F", "commodity"), ("ZS=F", "commodity"), ("KC=F", "commodity"),
    ("SB=F", "commodity"), ("CC=F", "commodity"), ("CT=F", "commodity"),
    ("^GSPC", "índice"), ("^IXIC", "índice"), ("^BVSP", "índice"),
    ("EURUSD=X", "fx"), ("BRL=X", "fx"),
    ("AAL", "ação"), ("ZIM", "ação"),
]

_RANK = {"rows": {}, "ts": 0.0, "building": False, "done": 0}
_RANK_LOCK = threading.Lock()
_EVENTS = []  # mudanças de sinal/divergências novas entre builds do ranking


def pop_events():
    """Consome os eventos acumulados (usado pelo watcher de alertas)."""
    global _EVENTS
    with _RANK_LOCK:
        events, _EVENTS = _EVENTS, []
        return events


def ranking(ttl=900):
    """Snapshot ranqueado do universo; constrói em background quando vence."""
    with _RANK_LOCK:
        if time.time() - _RANK["ts"] > ttl and not _RANK["building"]:
            _RANK["building"] = True
            _RANK["done"] = 0
            threading.Thread(target=_build_ranking, daemon=True).start()
        rows = sorted(_RANK["rows"].values(),
                      key=lambda r: (r.get("score") is None, -(r.get("score") or 0)))
        return {"rows": rows, "building": _RANK["building"],
                "done": _RANK["done"], "total": len(UNIVERSE),
                "ts": int(_RANK["ts"] * 1000) if _RANK["ts"] else None}


def _track_signal(sym, sig):
    """Grava o 1º sinal do dia (UTC) por símbolo — auditoria forward sem
    reescrita: o que foi emitido fica registrado como foi emitido."""
    try:
        if sig.get("price") is None:
            return
        from datetime import datetime, timezone
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with sqlite3.connect(_TRACK_DB) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS signals(
                symbol TEXT NOT NULL, date TEXT NOT NULL, label TEXT,
                score REAL, confidence REAL, price REAL, reasons TEXT,
                ts INTEGER, PRIMARY KEY(symbol, date))""")
            c.execute("INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,?,?)",
                      (sym, day, sig["label"], sig["score"], sig["confidence"],
                       sig["price"], json.dumps(sig["reasons"], ensure_ascii=False),
                       int(time.time() * 1000)))
    except Exception:
        pass  # tracking nunca derruba o ranking


def _build_ranking():
    def one(item):
        sym, klass = item
        try:
            r = analyze(sym)
            sig = r["signal"]
            _track_signal(sym, sig)
            row = {"symbol": sym, "class": klass, "label": sig["label"],
                   "score": sig["score"], "confidence": sig["confidence"],
                   "coverage_pct": sig["coverage_pct"], "price": sig["price"],
                   "n_factors": len(sig["reasons"]),
                   "divergences": [d["title"] for d in r["divergences"]]}
        except Exception as e:
            row = {"symbol": sym, "class": klass, "label": "ERRO",
                   "score": None, "error": str(e)[:80]}
        with _RANK_LOCK:
            _RANK["rows"][sym] = row
            _RANK["done"] += 1

    with _RANK_LOCK:
        old = dict(_RANK["rows"])  # baseline do build anterior (vazio no 1º)
    try:
        from concurrent.futures import ThreadPoolExecutor
        # ponytail: pool pequeno para não estourar rate limit do CoinGecko/Bybit
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(one, UNIVERSE))
        with _RANK_LOCK:
            for sym, new in _RANK["rows"].items():
                prev = old.get(sym)
                if not prev or new.get("score") is None or prev.get("score") is None:
                    continue  # sem baseline (1º build/erro) não há transição a alertar
                if new["label"] != prev["label"]:
                    _EVENTS.append({"symbol": sym, "kind": "signal_change",
                                    "detail": f"{prev['label']} → {new['label']} "
                                              f"(score {prev['score']:+.1f} → {new['score']:+.1f})"})
                for d in set(new.get("divergences") or []) - set(prev.get("divergences") or []):
                    _EVENTS.append({"symbol": sym, "kind": "divergence_new", "detail": d})
    finally:
        with _RANK_LOCK:
            _RANK["ts"] = time.time()
            _RANK["building"] = False

_AIRLINES = {"AAL", "DAL", "UAL", "LUV", "ALK", "JBLU", "BKNG", "ABNB",
             "MAR", "HLT", "EXPE", "RCL", "CCL", "NCLH"}
_SHIPPING = {"ZIM", "MAERSK-B.CO", "HLAG.DE", "FRO", "STNG", "BDRY", "GOGL",
             "SBLK", "DAC", "GSL"}
_AGRI = {"KC=F": "café", "CC=F": "cacau", "SB=F": "açúcar", "ZC=F": "milho",
         "ZS=F": "soja", "ZW=F": "trigo", "CT=F": "algodão"}


def analyze(symbol="BTC"):
    sym = (symbol or "BTC").upper().replace("-USD", "")
    return _cached(("analyze", sym), 900, lambda: _full(symbol, sym))


def tracking(ttl=3600):
    """Auto-auditoria forward: sinais gravados × retorno realizado 7/30d."""
    return _cached("tracking", ttl, _tracking)


def _tracking():
    import pandas as pd
    import yfinance as yf
    from providers.onchain_data import _CG_IDS
    rows = []
    if os.path.exists(_TRACK_DB):
        with sqlite3.connect(_TRACK_DB) as c:
            c.row_factory = sqlite3.Row
            try:
                rows = [dict(r) for r in c.execute(
                    "SELECT symbol,date,label,score,confidence,price,ts "
                    "FROM signals ORDER BY date, symbol")]
            except sqlite3.OperationalError:
                rows = []
    method = ("1 sinal/símbolo/dia (o 1º emitido, nunca reescrito); retorno = "
              "fechamento no 1º pregão ≥ D+7/D+30 sobre o preço no momento do "
              "sinal; favorável: COMPRA sobe, VENDA cai")
    if not rows:
        return {"signals": [], "summary": {}, "method": method,
                "note": "gravação iniciada — os primeiros retornos maturam em 7 dias",
                "ts": int(time.time() * 1000)}

    closes = {}
    for sym in {r["symbol"] for r in rows}:
        if sym in _CG_IDS:
            yf_sym = f"{sym}-USD"
        else:
            try:
                from providers import tradfi_data
                yf_sym = tradfi_data.resolve(sym)
            except Exception:
                yf_sym = sym
        try:
            h = yf.Ticker(yf_sym).history(period="1y", interval="1d",
                                          auto_adjust=True)["Close"].dropna()
            h.index = pd.to_datetime(h.index, utc=True).tz_convert(None).normalize()
            closes[sym] = h
        except Exception:
            pass

    for r in rows:
        s = closes.get(r["symbol"])
        for d in (7, 30):
            r[f"fwd_{d}d_pct"] = None
            if s is None or not r["price"]:
                continue
            target = pd.Timestamp(r["date"]) + pd.Timedelta(days=d)
            fut = s[s.index >= target]
            if len(fut):    # só matura quando o pregão D+N existe
                r[f"fwd_{d}d_pct"] = round((float(fut.iloc[0]) / r["price"] - 1) * 100, 2)

    summary = {}
    for label in ("COMPRA", "VENDA", "NEUTRO"):
        sub = [r for r in rows if r["label"] == label]
        entry = {"signals": len(sub)}
        for d in (7, 30):
            m = [r[f"fwd_{d}d_pct"] for r in sub if r[f"fwd_{d}d_pct"] is not None]
            entry[f"matured_{d}d"] = len(m)
            entry[f"avg_fwd_{d}d_pct"] = round(sum(m) / len(m), 2) if m else None
            if m and label in ("COMPRA", "VENDA"):
                fav = [x > 0 for x in m] if label == "COMPRA" else [x < 0 for x in m]
                entry[f"favorable_{d}d_pct"] = round(sum(fav) / len(fav) * 100, 1)
            else:
                entry[f"favorable_{d}d_pct"] = None
        summary[label] = entry
    return {"signals": rows[-120:][::-1], "summary": summary, "method": method,
            "ts": int(time.time() * 1000)}


def _driver_closes():
    """Fechamentos 6m dos drivers macro (SPX, DXY, WTI, BTC)."""
    return _cached("drivers", 3600, _fetch_driver_closes)


def _fetch_driver_closes():
    import pandas as pd
    import yfinance as yf
    out = {}
    for name, t in (("SPX", "^GSPC"), ("DXY", "DX-Y.NYB"),
                    ("petróleo WTI", "CL=F"), ("BTC", "BTC-USD")):
        try:
            h = yf.Ticker(t).history(period="6mo", interval="1d",
                                     auto_adjust=True)["Close"].dropna()
            h.index = pd.to_datetime(h.index, utc=True).tz_convert(None).normalize()
            out[name] = h
        except Exception:
            pass
    return out


def _fwd_stats(close, mask, days=21):
    """Retorno médio e taxa de acerto N dias à frente quando a condição valeu."""
    fwd = (close.shift(-days) / close - 1).where(mask).dropna()
    if len(fwd) < 20:
        return None
    return {"samples": int(mask.sum()), "valid_30d": int(len(fwd)),
            "return_30d_pct": round(float(fwd.mean() * 100), 2),
            "win_rate_30d_pct": round(float((fwd > 0).mean() * 100), 1),
            "return_90d_pct": None}


def _full(raw, sym):
    import pandas as pd
    import yfinance as yf
    from providers.technical_data import _sma, _rsi
    from providers.onchain_data import _CG_IDS

    crypto = sym in _CG_IDS or "-USD" in (raw or "").upper()
    is_btc = sym == "BTC"
    if crypto:
        yf_sym = f"{sym}-USD"
    else:
        try:
            from providers import tradfi_data
            yf_sym = tradfi_data.resolve(sym)
        except Exception:
            yf_sym = sym

    factors, reasons, divergences, health = [], [], [], []
    validation, insights = {}, []
    close = price = ret30 = None
    trend_up = None
    date = None
    liquidity = {"open_interest_usd": None, "pi_top_distance_pct": None}

    def source(name, status, detail=None):
        health.append({"source": name, "status": status,
                       **({"detail": str(detail)[:80]} if detail else {})})

    def vote(v, reason):
        factors.append(v)
        reasons.append(reason)

    # BTC: modelo on-chain do research (já cruza on-chain × técnica)
    if is_btc:
        try:
            from providers.btc_onchain_metrics import payload
            from research.analyze_btc_onchain_signals import (
                build_frame, score_history, summarize)
            data = payload()
            frame = score_history(build_frame(data))
            summary = summarize(frame)
            ready = frame[frame.model_ready]
            if ready.empty:
                raise ValueError("métricas sem histórico comum suficiente")
            last = ready.iloc[-1]
            price = float(last.price)
            date = str(ready.index[-1].date())
            vote({"COMPRA": 2, "VENDA": -2}.get(last.signal, 0),
                 f"Modelo on-chain+técnica (backtest causal): {last.signal} "
                 f"score {float(last.score):+.2f} — {last.reasons}")
            for k, v in summary["historical"].items():
                validation[f"On-chain: {k}"] = {
                    "samples": v["events"], "valid_30d": v["valid_30d"],
                    "return_30d_pct": v["avg_forward_30d_pct"],
                    "win_rate_30d_pct": v["favorable_30d_pct"],
                    "return_90d_pct": v["avg_forward_90d_pct"]}
            p = frame.price
            for col, lbl, mode in (("nupl", "NUPL", "abs"),
                                   ("sth_mvrv", "STH-MVRV", "one"),
                                   ("sth_sopr", "STH-SOPR", "one")):
                if col not in frame or frame[col].dropna().empty:
                    continue
                pchg = p.pct_change(30, fill_method=None).iloc[-1]
                mchg = (frame[col].diff(30) if mode == "abs"
                        else (frame[col] - 1).diff(30)).iloc[-1]
                thr = .05 if mode == "abs" else .03
                if pchg > 0.05 and mchg < -thr:
                    divergences.append({"severity": "warning",
                                        "title": f"Preço ↑ / {lbl} deteriorando",
                                        "detail": f"30d: preço {pchg*100:.1f}%, métrica {mchg:+.3f}"})
                if pchg < -0.05 and mchg > thr:
                    divergences.append({"severity": "opportunity",
                                        "title": f"Preço ↓ / {lbl} melhorando",
                                        "detail": f"30d: preço {pchg*100:.1f}%, métrica {mchg:+.3f}"})
            ret30 = float(p.pct_change(30, fill_method=None).iloc[-1] * 100)
            trend_up = bool(last.price > frame.price.rolling(200).mean().iloc[-1])
            liquidity["open_interest_usd"] = (data.get("open_interest") or {}).get("total_usd")
            liquidity["pi_top_distance_pct"] = float(
                (last.price / frame.pi_350dma_x2.iloc[-1] - 1) * 100)
            n_series = len(data.get("series") or {})
            source("On-chain BTC (bitcoin-data/Glassnode + Binance)", "ok",
                   f"{n_series} séries; {len(data.get('unavailable') or [])} indisponíveis")
        except Exception as e:
            source("On-chain BTC", "error", e)

    # técnica (yfinance) — para BTC o research já vota; aqui só contexto
    try:
        h = yf.Ticker(yf_sym).history(period="5y", interval="1d", auto_adjust=True)
        yclose = h["Close"].dropna()
        if len(yclose) < 220:
            raise ValueError(f"histórico insuficiente ({len(yclose)} pregões)")
        close = yclose
        source(f"Yahoo Finance ({yf_sym})", "ok")
    except Exception as e:
        source(f"Yahoo Finance ({yf_sym})", "error", e)

    if close is not None and not is_btc:
        price = float(close.iloc[-1])
        date = str(close.index[-1].date())
        sma200 = _sma(close, 200)
        rsi_series = _rsi(close)
        rsi = float(rsi_series.iloc[-1])
        ret30 = float(close.pct_change(21, fill_method=None).iloc[-1] * 100)
        above = close > sma200
        trend_up = bool(above.iloc[-1])

        vote(1 if trend_up else -1,
             f"Tendência: preço {'acima' if trend_up else 'abaixo'} da SMA200 "
             f"({(price / float(sma200.iloc[-1]) - 1) * 100:+.1f}%)")
        s = _fwd_stats(close, (above if trend_up else ~above) & sma200.notna())
        if s:
            validation["Tendência atual (fwd 30d)"] = s

        vote(1 if ret30 > 5 else -1 if ret30 < -5 else 0, f"Momentum 30d: {ret30:+.1f}%")
        vote(1 if rsi < 30 else -1 if rsi > 70 else 0,
             f"RSI(14): {rsi:.0f}"
             + (" — sobrevendido" if rsi < 30 else " — sobrecomprado" if rsi > 70 else ""))
        if rsi > 70 or rsi < 30:
            s = _fwd_stats(close, rsi_series > 70 if rsi > 70 else rsi_series < 30)
            if s:
                validation[f"RSI {'>70' if rsi > 70 else '<30'} (fwd 30d)"] = s
        if rsi > 70 and ret30 > 5:
            divergences.append({"severity": "warning",
                                "title": "Preço esticado com RSI sobrecomprado",
                                "detail": f"30d {ret30:+.1f}%, RSI {rsi:.0f}"})
        if rsi < 30 and ret30 < -5:
            divergences.append({"severity": "opportunity",
                                "title": "Queda forte com RSI sobrevendido",
                                "detail": f"30d {ret30:+.1f}%, RSI {rsi:.0f}"})

    # sazonalidade do mês corrente
    try:
        from providers import seasonality_data
        sea = seasonality_data.analyze(yf_sym)
        month = pd.Timestamp.now("UTC").month
        m = next(x for x in sea["monthly_stats"] if x["month"] == month)
        if m["avg_pct"] is not None and m["samples"] >= 5:
            win = m["win_rate_pct"] or 50
            vote(1 if m["avg_pct"] > 0 and win >= 55
                 else -1 if m["avg_pct"] < 0 and win <= 45 else 0,
                 f"Sazonalidade do mês: média {m['avg_pct']:+.2f}%, "
                 f"acerto {win:.0f}% em {m['samples']} anos")
            validation["Sazonalidade (mês atual)"] = {
                "samples": m["samples"], "valid_30d": m["samples"],
                "return_30d_pct": m["avg_pct"], "win_rate_30d_pct": win,
                "return_90d_pct": None}
            if trend_up is True and win <= 40:
                divergences.append({"severity": "warning",
                                    "title": "Tendência de alta em mês historicamente fraco",
                                    "detail": f"acerto histórico de {win:.0f}% no mês {month}"})
            if trend_up is False and win >= 60:
                divergences.append({"severity": "opportunity",
                                    "title": "Tendência de baixa em mês historicamente forte",
                                    "detail": f"acerto histórico de {win:.0f}% no mês {month}"})
        source("Sazonalidade (Yahoo)", "ok")
    except Exception as e:
        source("Sazonalidade", "error", e)

    # macro: liquidez Fed + regime de risco (vale para qualquer ativo)
    try:
        from providers import liquidity_data
        liq = {r["id"]: r for r in liquidity_data.snapshot()["series"]
               if r.get("status") == "ok"}
        if all(k in liq for k in ("fed_balance_sheet", "reverse_repo", "treasury_account")):
            # net liquidity em USD bi: WALCL(mi) - RRP(bi) - TGA(mi)
            d30 = (liq["fed_balance_sheet"]["change_30"] / 1000
                   - liq["reverse_repo"]["change_30"]
                   - liq["treasury_account"]["change_30"] / 1000)
            vote(1 if d30 > 25 else -1 if d30 < -25 else 0,
                 f"Liquidez líquida Fed (WALCL−RRP−TGA): {d30:+,.0f} USD bi em 30 obs")
        hy, vix = liq.get("high_yield_spread"), liq.get("vix")
        if hy and vix:
            risk = (1 if hy["change_30"] < -0.10 else -1 if hy["change_30"] > 0.30 else 0) \
                 + (1 if vix["value"] < 14 else -1 if vix["value"] > 25 else 0)
            vote(1 if risk > 0 else -1 if risk < 0 else 0,
                 f"Regime de risco: HY spread {hy['value']:.2f}% ({hy['change_30']:+.2f} em 30 obs), "
                 f"VIX {vix['value']:.1f}")
            if trend_up is True and risk < 0:
                divergences.append({"severity": "warning",
                                    "title": "Alta do ativo contra aperto macro",
                                    "detail": f"HY spread {hy['change_30']:+.2f}, VIX {vix['value']:.1f}"})
        source("FRED (liquidez/risco)", "ok")
    except Exception as e:
        source("FRED (liquidez/risco)", "error", e)

    # cripto: funding, TVL, Fear & Greed, amplitude, top traders, dev
    if crypto:
        try:
            from providers import onchain_data
            c = onchain_data.coin(sym)
            d = c.get("deriv") or {}
            f = d.get("funding_pct")
            if liquidity["open_interest_usd"] is None:
                liquidity["open_interest_usd"] = d.get("oi_usd")
            if f is not None:
                vote(1 if f < 0 else -1 if f >= 0.03 else 0,
                     f"Funding perp Bybit: {f:+.4f}%/8h"
                     + (" — shorts pagando (contrário altista)" if f < 0
                        else " — longs lotados (contrário baixista)" if f >= 0.03 else ""))
                if ret30 is not None:
                    if f >= 0.03 and ret30 > 10:
                        divergences.append({"severity": "warning",
                                            "title": "Alta sustentada por alavancagem comprada cara",
                                            "detail": f"funding {f:+.4f}%/8h, preço {ret30:+.1f}% em 30d"})
                    if f < 0 and ret30 < -10:
                        divergences.append({"severity": "opportunity",
                                            "title": "Queda com shorts pagando funding",
                                            "detail": f"funding {f:+.4f}%/8h, preço {ret30:+.1f}% em 30d"})
            tvl = c.get("tvl") or {}
            if tvl.get("chg_30d_pct") is not None:
                t30 = tvl["chg_30d_pct"]
                vote(1 if t30 > 10 else -1 if t30 < -10 else 0,
                     f"TVL da chain {tvl.get('chain')}: {t30:+.1f}% em 30d (DeFiLlama)")
                if ret30 is not None and ret30 > 5 and t30 < -10:
                    divergences.append({"severity": "warning",
                                        "title": "Preço subindo com TVL da chain encolhendo",
                                        "detail": f"preço {ret30:+.1f}% vs TVL {t30:+.1f}% em 30d"})
            dev = (c.get("profile") or {}).get("dev") or {}
            if dev.get("commits_4w") == 0:
                divergences.append({"severity": "warning",
                                    "title": "Atividade de desenvolvimento zerada (4 semanas)",
                                    "detail": "CoinGecko developer data — projeto pode estar abandonado"})
            errs = c.get("errors") or []
            source("CoinGecko/Bybit (cripto)", "ok" if not errs else "partial",
                   "; ".join(errs) if errs else None)
        except Exception as e:
            source("CoinGecko/Bybit (cripto)", "error", e)

        try:
            from providers.onchain_data import _get
            fng = _get("https://api.alternative.me/fng/?limit=1")["data"][0]
            v = int(fng["value"])
            vote(1 if v <= 25 else -1 if v >= 75 else 0,
                 f"Fear & Greed: {v} ({fng.get('value_classification')}) — leitura contrária")
            source("Fear & Greed (alternative.me)", "ok")
        except Exception as e:
            source("Fear & Greed", "error", e)

        try:
            from providers import altdata
            cm = altdata.crypto_micro()
            pp = cm.get("pct_positive")
            if pp is not None:
                vote(1 if pp <= 40 else -1 if pp >= 80 else 0,
                     f"Amplitude de funding: {pp:.0f}% dos {cm['n_perps']} perps positivos"
                     + (" — mercado inteiro alavancado comprado" if pp >= 80 else ""))
            source("Bybit (amplitude de funding)", "ok")
        except Exception as e:
            source("Bybit (amplitude de funding)", "error", e)

        try:
            from providers import insider_data
            sm = insider_data.smart_money(sym)
            feed = next((x for x in sm["feeds"] if x.get("kind") == "smart_money_proxy"), None)
            if feed and feed.get("latest"):
                ratio = feed["latest"]["ratio"]
                vote(1 if ratio >= 1.5 else -1 if ratio <= 0.7 else 0,
                     f"Top traders Binance (proxy): long/short {ratio:.2f}")
                source("Binance top traders", "ok")
        except Exception as e:
            source("Binance top traders", "error", e)

    # ações EUA: insiders SEC; aéreas: TSA; frete: GSCPI
    is_us_stock = not crypto and "." not in sym and "=" not in sym and "^" not in sym
    if is_us_stock:
        try:
            from providers import insider_data
            sm = insider_data.smart_money(sym)
            sec = next((x for x in sm["feeds"] if x.get("source") == "SEC EDGAR"), None)
            if sec:
                cutoff = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                recent = [x for x in sec["filings"] if x["date"] >= cutoff]
                # filings não dizem compra/venda sem parsear cada doc — insight, não voto
                insights.append(f"{len(recent)} filings de insiders (Forms 3/4/5) nos "
                                f"últimos 30d — detalhe na aba Insiders")
                source("SEC EDGAR (insiders)", "ok")
        except Exception as e:
            source("SEC EDGAR (insiders)", "error", e)

        # opções: put/call OI (contrário nos extremos) + IV vs vol realizada
        try:
            from providers import markets_data
            ch = markets_data.option_chain(sym)
            if not ch.get("error") and ch.get("dte", 0) < 7:
                # vencimento 0DTE distorce IV e OI — usa o 1º com >=7 dias
                from datetime import date as _date
                today_d = _date.today()
                exp = next((e for e in ch["expiries"]
                            if (_date.fromisoformat(e) - today_d).days >= 7), None)
                if exp:
                    ch = markets_data.option_chain(sym, exp)
            if not ch.get("error"):
                put_oi = sum(r.get("oi") or 0 for r in ch["puts"])
                call_oi = sum(r.get("oi") or 0 for r in ch["calls"])
                if call_oi:
                    pcr = put_oi / call_oi
                    vote(1 if pcr >= 1.5 else -1 if pcr <= 0.5 else 0,
                         f"Opções: put/call OI {pcr:.2f} no venc. {ch['expiry']}"
                         + (" — pessimismo lotado (contrário altista)" if pcr >= 1.5
                            else " — otimismo lotado (contrário baixista)" if pcr <= 0.5 else ""))
                iv, hv = ch.get("atm_iv"), ch.get("hist_vol30")
                # Yahoo às vezes devolve IV ~0 (dado velho) — só vota se plausível
                if iv and hv and hv > 0 and iv >= 5:
                    vote(1 if iv / hv >= 1.5 else 0,
                         f"IV ATM {iv:.0f}% vs realizada 30d {hv:.0f}% ({iv / hv:.1f}x)"
                         + (" — medo pago caro (contrário)" if iv / hv >= 1.5 else ""))
                # movimento implícito pelo PREÇO do straddle ATM (robusto a IV ruim)
                spot = ch.get("spot")
                if spot:
                    def _atm_px(side):
                        rows_ = [r for r in side if r.get("strike") and
                                 (r.get("last") or (r.get("bid") and r.get("ask")))]
                        if not rows_:
                            return None
                        r = min(rows_, key=lambda r: abs(r["strike"] - spot))
                        if abs(r["strike"] - spot) / spot > 0.05:
                            return None
                        return (r["bid"] + r["ask"]) / 2 if r.get("bid") and r.get("ask") else r["last"]
                    c_px, p_px = _atm_px(ch["calls"]), _atm_px(ch["puts"])
                    if c_px and p_px:
                        insights.append(f"o straddle ATM implica ±{(c_px + p_px) / spot * 100:.1f}% "
                                        f"até {ch['expiry']} ({ch['dte']}d)")
                source("Opções (Yahoo)", "ok")
            else:
                source("Opções (Yahoo)", "partial", ch["error"])
        except Exception as e:
            source("Opções (Yahoo)", "error", e)

        # short interest: extremo = combustível de squeeze (contrário altista)
        try:
            info = yf.Ticker(yf_sym).info or {}
            spf = info.get("shortPercentOfFloat")
            if spf is not None:
                spf_pct = spf * 100 if spf <= 1 else spf
                sr = info.get("shortRatio")
                vote(1 if spf_pct >= 15 else 0,
                     f"Short interest: {spf_pct:.1f}% do float"
                     + (f" · {sr:.1f} dias para cobrir" if sr else "")
                     + (" — short lotado (combustível de squeeze)" if spf_pct >= 15 else ""))
                source("Short interest (Yahoo)", "ok")
        except Exception as e:
            source("Short interest (Yahoo)", "error", e)

    # earnings próximos: evento de vol — insight, não voto
    if not crypto and "=" not in yf_sym and "^" not in yf_sym:
        try:
            cal = yf.Ticker(yf_sym).calendar or {}
            dates = cal.get("Earnings Date") or []
            today = pd.Timestamp.now().date()
            nxt = next((d for d in sorted(dates) if d >= today), None)
            if nxt and (nxt - today).days <= 21:
                insights.append(f"earnings em {(nxt - today).days} dias ({nxt}) — "
                                "evento de vol: sinais técnico e sazonal valem menos até lá")
        except Exception:
            pass  # calendário é opcional; sem earnings não é erro de fonte

    if sym in _AIRLINES:
        try:
            from providers import altdata
            tsa = altdata.traffic()
            mom = tsa.get("mom_pct")
            if mom is not None:
                vote(1 if mom >= 2 else -1 if mom <= -2 else 0,
                     f"Tráfego aéreo TSA: {mom:+.1f}% vs mês anterior (média 7d)")
            source("TSA (tráfego aéreo)", "ok")
        except Exception as e:
            source("TSA (tráfego aéreo)", "error", e)

    if sym in _SHIPPING:
        try:
            from providers import altdata
            gs = altdata.gscpi_series()
            last_gs = gs["values"][-1]
            yoy = last_gs - gs["values"][-13] if len(gs["values"]) >= 13 else None
            if yoy is not None:
                vote(1 if yoy > 0.3 else -1 if yoy < -0.3 else 0,
                     f"GSCPI (pressão de supply chain): {last_gs:+.2f} ({yoy:+.2f} vs 1 ano) — "
                     "pressão subindo tende a sustentar fretes")
            source("NY Fed GSCPI", "ok")
        except Exception as e:
            source("NY Fed GSCPI", "error", e)

    # commodities: COT (CFTC); agrícolas: ENSO + clima
    if not crypto and yf_sym.endswith("=F"):
        try:
            from providers import insider_data
            sm = insider_data.smart_money(yf_sym)
            cot = next((x for x in sm["feeds"] if x.get("kind") == "positioning"), None)
            if cot and cot.get("latest"):
                pctl = cot["latest"].get("net_percentile_2y")
                if pctl is not None:
                    vote(1 if pctl <= 10 else -1 if pctl >= 90 else 0,
                         f"COT (CFTC): net especulativo no percentil {pctl:.0f}% de 2 anos"
                         + (" — posicionamento lotado (contrário)" if pctl >= 90 or pctl <= 10 else ""))
                    if pctl >= 90 and trend_up is True:
                        divergences.append({"severity": "warning",
                                            "title": "Alta com especuladores lotados (COT)",
                                            "detail": f"net não-comercial no percentil {pctl:.0f}%"})
            source("CFTC COT", "ok" if cot else "partial")
        except Exception as e:
            source("CFTC COT", "error", e)

    crop = _AGRI.get(yf_sym)
    if crop:
        try:
            from providers import altdata
            cli = altdata.climate()
            enso = cli["enso"]
            alerts = [f"{r['region']}: {', '.join(r['flags'])}"
                      for r in ((cli.get("regions") or {}).get("rows") or [])
                      if r.get("flags") and crop in (r.get("crops") or "").lower()]
            active = enso["status"] != "Neutro"
            vote(1 if (active or alerts) else 0,
                 f"Clima: ENSO {enso['status']} (anomalia {enso['anom']:+.1f}°C)"
                 + (f"; alertas em regiões de {crop}: {'; '.join(alerts)}" if alerts
                    else " — risco de oferta" if active else " — sem sinal dominante"))
            if alerts:
                divergences.append({"severity": "opportunity",
                                    "title": f"Estresse climático em região produtora de {crop}",
                                    "detail": "; ".join(alerts)[:140]})
            source("NOAA ENSO + met.no", "ok")
        except Exception as e:
            source("NOAA ENSO + met.no", "error", e)

    # drivers: com quem o ativo anda (correlação 60d) — educativo
    if close is not None:
        try:
            c2 = close.copy()
            c2.index = pd.to_datetime(c2.index, utc=True).tz_convert(None).normalize()
            rets = c2.pct_change()
            skip = {"BTC": "BTC", "CL=F": "petróleo WTI", "^GSPC": "SPX"}.get(sym)
            hits = []
            for name, s in _driver_closes().items():
                if name == skip:
                    continue
                pair = pd.concat([rets, s.pct_change()], axis=1,
                                 join="inner").dropna().tail(60)
                if len(pair) < 40:
                    continue
                cv = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                if abs(cv) >= 0.5:
                    hits.append(f"{'junto com' if cv > 0 else 'contra'} {name} ({cv:+.2f})")
            insights.append(("drivers 60d: move " + "; ".join(hits)) if hits else
                            ("drivers 60d: sem correlação dominante (|corr| < 0.5 "
                             "com SPX/DXY/WTI/BTC) — movimento é idiossincrático"))
        except Exception:
            pass  # insight opcional

    # consolidação
    if not factors:
        raise ValueError(f"nenhuma fonte disponível para {sym}")
    score = sum(factors)
    n = len(factors)
    label = "COMPRA" if score >= 2 else "VENDA" if score <= -2 else "NEUTRO"
    ok_sources = sum(1 for h in health if h["status"] == "ok")
    reasons += insights
    return {"symbol": sym,
            "signal": {"label": label, "score": float(score),
                       "confidence": round(abs(score) / n * 100, 1),
                       "confidence_basis": "concordância entre os fatores disponíveis",
                       "coverage_pct": round(ok_sources / len(health) * 100, 1),
                       "reasons": reasons, "price": price, "date": date},
            "validation": validation, "divergences": divergences, "health": health,
            "liquidity": liquidity, "ts": int(time.time() * 1000)}


if __name__ == "__main__":
    for s in ("BTC", "ETH", "PETR4.SA", "AAL", "KC=F", "ZIM"):
        r = analyze(s)
        sig = r["signal"]
        assert sig["label"] in ("COMPRA", "VENDA", "NEUTRO") and len(sig["reasons"]) >= 3, r
        print(f"{s}: {sig['label']} score={sig['score']:+.1f} conf={sig['confidence']}% "
              f"cobertura={sig['coverage_pct']}% | fatores={len(sig['reasons'])} "
              f"divergências={len(r['divergences'])} validações={len(r['validation'])}")
