"""Snapshot de liquidez US/sistêmica via CSV público do FRED."""
from __future__ import annotations

import time
from io import StringIO

import pandas as pd
import requests

from common import ttl_cache

_cached = ttl_cache()

# id interno -> (série FRED, rótulo, unidade)
SERIES = {
    "fed_balance_sheet": ("WALCL", "Fed balance sheet", "USD millions"),
    "reverse_repo": ("RRPONTSYD", "Reverse repo", "USD billions"),
    "treasury_account": ("WTREGEN", "Treasury General Account", "USD millions"),
    "high_yield_spread": ("BAMLH0A0HYM2", "US high-yield spread", "percent"),
    "vix": ("VIXCLS", "VIX", "index"),
    "broad_dollar": ("DTWEXBGS", "Broad dollar index", "index"),
}

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _series_row(key: str, sid: str, label: str, unit: str) -> dict:
    r = requests.get(FRED_CSV.format(sid), timeout=25)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
    return {
        "id": key, "series": sid, "label": label, "unit": unit, "status": "ok",
        "value": float(vals.iloc[-1]),
        # 30 observações atrás, ou a mais antiga disponível
        "change_30": float(vals.iloc[-1] - vals.iloc[-min(31, len(vals))]),
    }


def _snapshot() -> dict:
    rows = []
    for key, (sid, label, unit) in SERIES.items():
        try:
            rows.append(_series_row(key, sid, label, unit))
        except Exception as e:
            # Uma série fora do ar não derruba o painel inteiro.
            rows.append({"id": key, "series": sid, "label": label,
                         "status": "error", "error": str(e)[:80]})
    return {"series": rows, "source": "Federal Reserve Economic Data (FRED)",
            "ts": int(time.time() * 1000)}


def snapshot() -> dict:
    return _cached("snapshot", 3600, _snapshot)
