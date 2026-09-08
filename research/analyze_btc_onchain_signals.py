"""Gera sinais históricos de BTC a partir das métricas on-chain do Graph.

Uso:
    py -3 research/analyze_btc_onchain_signals.py
    py -3 research/analyze_btc_onchain_signals.py --output data/btc_onchain_signals.csv

O score é causal: médias e percentis usam somente valores conhecidos até cada
data. O resultado é pesquisa quantitativa, não recomendação financeira.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.btc_onchain_metrics import payload  # noqa: E402


LABELS = {1: "COMPRA", 0: "NEUTRO", -1: "VENDA"}


def _add_series(frame: pd.DataFrame, name: str, series: dict | None) -> pd.DataFrame:
    if not series or not series.get("ts"):
        return frame
    s = pd.Series(series["values"], index=pd.to_datetime(series["ts"], unit="ms", utc=True),
                  name=name, dtype="float64")
    s.index = s.index.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return frame.join(s, how="outer")


def build_frame(data: dict) -> pd.DataFrame:
    pi = data.get("pi_cycle") or {}
    if not pi.get("ts"):
        raise ValueError("histórico de preço/Pi Cycle indisponível")
    idx = pd.to_datetime(pi["ts"], unit="ms", utc=True).normalize()
    df = pd.DataFrame({"price": pi["price"], "pi_111dma": pi["dma111"],
                       "pi_350dma_x2": pi["dma350x2"]}, index=idx)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for name, series in (data.get("series") or {}).items():
        df = _add_series(df, name, series)
    return df.sort_index().ffill(limit=3)


def _expanding_quantile(s: pd.Series, q: float, minimum=180) -> pd.Series:
    # shift(1) guarantees today's value never participates in today's threshold.
    return s.shift(1).expanding(min_periods=minimum).quantile(q)


def score_history(df: pd.DataFrame, buy_threshold=1.5, sell_threshold=-1.5) -> pd.DataFrame:
    out = df.copy()
    contributions: dict[str, pd.Series] = {}
    availability: dict[str, pd.Series] = {}

    def add(name, condition_buy, condition_sell, weight=1.0, available=None):
        contributions[name] = pd.Series(
            np.select([condition_buy.fillna(False), condition_sell.fillna(False)],
                      [weight, -weight], default=0.0), index=out.index)
        availability[name] = (available if available is not None else
                              pd.Series(True, index=out.index)).fillna(False)

    if "nupl" in out:
        # Regime thresholds widely used for NUPL: distress < 0, euphoria > 0.7.
        add("NUPL", out.nupl < 0.0, out.nupl > 0.70, 1.5, out.nupl.notna())
    if "sth_mvrv" in out:
        hi = _expanding_quantile(out.sth_mvrv, .90)
        add("STH-MVRV", out.sth_mvrv < 0.95, out.sth_mvrv > hi, 1.25,
            out.sth_mvrv.notna() & hi.notna())
    if "sth_sopr" in out:
        sopr7 = out.sth_sopr.rolling(7, min_periods=4).mean()
        hi = _expanding_quantile(sopr7, .90)
        add("STH-SOPR", sopr7 < 0.985, sopr7 > hi, 0.75,
            sopr7.notna() & hi.notna())
    if "exchange_whale_ratio" in out:
        lo, hi = (_expanding_quantile(out.exchange_whale_ratio, q) for q in (.10, .90))
        add("Whale Ratio", out.exchange_whale_ratio < lo, out.exchange_whale_ratio > hi, 1.0,
            out.exchange_whale_ratio.notna() & lo.notna() & hi.notna())
    if "estimated_leverage_ratio" in out:
        lo, hi = (_expanding_quantile(out.estimated_leverage_ratio, q) for q in (.10, .90))
        add("Leverage", out.estimated_leverage_ratio < lo,
            out.estimated_leverage_ratio > hi, .75,
            out.estimated_leverage_ratio.notna() & lo.notna() & hi.notna())
    if "retail_demand_30d" in out:
        lo, hi = (_expanding_quantile(out.retail_demand_30d, q) for q in (.10, .90))
        add("Retail 30D", out.retail_demand_30d < lo, out.retail_demand_30d > hi, .5,
            out.retail_demand_30d.notna() & lo.notna() & hi.notna())

    # ── técnica sobre o mesmo histórico de preço (cruza on-chain × técnica) ──
    # Peso subordinado (tilt): técnica segue tendência e on-chain é contrário;
    # com peso igual eles se cancelam exatamente nos topos/fundos que o modelo
    # existe para pegar. RSI é contrário e pode pesar mais.
    from providers.technical_data import _sma, _rsi
    sma200 = _sma(out.price, 200)
    add("Tendência SMA200", out.price > sma200, out.price < sma200, 0.25,
        sma200.notna())
    mom30 = out.price.pct_change(30, fill_method=None)
    add("Momentum 30d", mom30 > 0.10, mom30 < -0.10, 0.25, mom30.notna())
    rsi = _rsi(out.price)
    add("RSI(14)", rsi < 30, rsi > 70, 0.5, rsi.notna())

    # Pi Cycle: extreme discount is bullish; top-line proximity/crossover bearish.
    pi_top_ratio = out.price / out.pi_350dma_x2
    price_to_350dma = out.price / (out.pi_350dma_x2 / 2.0)
    add("Pi Cycle", price_to_350dma < .70,
        (out.pi_111dma >= out.pi_350dma_x2) | (pi_top_ratio > .95), 1.5,
        out.price.notna() & out.pi_111dma.notna() & out.pi_350dma_x2.notna())

    c = pd.DataFrame(contributions, index=out.index)
    # A historical score is only comparable after every configured component
    # has enough data (including expanding-quantile warm-up).
    ready = pd.DataFrame(availability, index=out.index).all(axis=1)
    out["model_ready"] = ready
    out["score"] = c.sum(axis=1).where(ready)
    out["signal_code"] = np.select(
        [out.score >= buy_threshold, out.score <= sell_threshold], [1, -1], default=0)
    out["signal"] = out.signal_code.map(LABELS)
    out.loc[~ready, "signal"] = "SEM_DADOS"
    out["reasons"] = ["; ".join(f"{k} {'+' if v > 0 else '-'}"
                                  for k, v in row.items() if v != 0) or "sem extremo"
                      for row in c.to_dict("records")]
    for days in (30, 90):
        out[f"forward_return_{days}d"] = out.price.shift(-days) / out.price - 1
    return out


def _independent_events(df: pd.DataFrame, label: str, min_gap_days=90) -> pd.DataFrame:
    transitions = df[(df.signal == label) & (df.signal.shift(1) != label)]
    chosen, last = [], None
    for idx in transitions.index:
        if last is None or (idx - last).days >= min_gap_days:
            chosen.append(idx)
            last = idx
    return df.loc[chosen]


def summarize(df: pd.DataFrame) -> dict:
    valid = df[df.model_ready] if "model_ready" in df else df
    summary = {"period": {"start": str(df.index.min().date()),
                           "end": str(df.index.max().date())},
               "latest": {"date": str(df.index[-1].date()),
                          "price": round(float(df.price.iloc[-1]), 2),
                          "score": round(float(df.score.iloc[-1]), 2),
                          "signal": df.signal.iloc[-1], "reasons": df.reasons.iloc[-1]},
               "historical": {}}
    for label in ("COMPRA", "VENDA", "NEUTRO"):
        rows = _independent_events(valid, label)
        r30 = rows.forward_return_30d.dropna()
        r90 = rows.forward_return_90d.dropna()
        favorable = (r30 > 0) if label != "VENDA" else (r30 < 0)
        def pct_stat(value):
            return None if pd.isna(value) else round(float(value) * 100, 2)
        summary["historical"][label] = {
            "events": len(rows), "days": len(rows),
            "valid_30d": len(r30), "valid_90d": len(r90),
            "avg_forward_30d_pct": pct_stat(r30.mean()),
            "positive_30d_pct": pct_stat((r30 > 0).mean()),
            "favorable_30d_pct": pct_stat(favorable.mean()),
            "avg_forward_90d_pct": pct_stat(r90.mean()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Sinais históricos on-chain do BTC")
    parser.add_argument("--output", default=str(ROOT / "data" / "btc_onchain_signals.csv"))
    parser.add_argument("--buy-threshold", type=float, default=1.5)
    parser.add_argument("--sell-threshold", type=float, default=-1.5)
    args = parser.parse_args()

    scored = score_history(build_frame(payload()), args.buy_threshold, args.sell_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index_label="date")
    report = summarize(scored)
    output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nCSV: {output}\nResumo: {output.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
