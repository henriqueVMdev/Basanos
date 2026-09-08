import numpy as np
import pandas as pd
from unittest.mock import patch

from providers.insider_data import _CFTC
from engine.portfolio_lab import _annualization_factor
from research.analyze_btc_onchain_signals import summarize
from providers.seasonality_data import _path
from providers.calendar_data import events
from providers import btc_onchain_metrics


def test_annualization_respects_shared_market_calendar():
    business = pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=520), data={"x": 0.0})
    crypto = pd.DataFrame(index=pd.date_range("2024-01-01", periods=730), data={"x": 0.0})
    assert 250 <= _annualization_factor(business) <= 262
    assert 364 <= _annualization_factor(crypto) <= 366


def test_signal_validation_counts_events_and_drops_missing_forward_returns():
    idx = pd.date_range("2020-01-01", periods=220, tz="UTC")
    df = pd.DataFrame(index=idx, data={"price": 100.0, "score": 1.0,
                                      "model_ready": True, "reasons": "test",
                                      "forward_return_30d": 0.1,
                                      "forward_return_90d": 0.2})
    df["signal"] = ["COMPRA"] * 100 + ["NEUTRO"] * 100 + ["VENDA"] * 20
    df.loc[idx[200], "forward_return_30d"] = np.nan
    result = summarize(df)["historical"]
    assert result["COMPRA"]["events"] == 1
    assert result["VENDA"]["events"] == 1
    assert result["VENDA"]["valid_30d"] == 0
    assert result["VENDA"]["favorable_30d_pct"] is None


def test_cftc_contracts_use_unique_exact_codes():
    codes = [value[1] for value in _CFTC.values()]
    assert len(codes) == len(set(codes))
    assert _CFTC["GC=F"][1] == "088691"


def test_seasonal_path_has_canonical_non_leap_calendar():
    idx = pd.to_datetime(["2020-02-28", "2020-02-29", "2020-03-01",
                          "2021-02-28", "2021-03-01"], utc=True)
    path = _path(pd.Series([.01, .50, .02, .01, .02], index=idx), 20)
    assert len(path) == 365
    # Feb-29 is excluded instead of shifting every subsequent calendar date.
    assert path[58] == 1.0
    assert path[59] == 3.02


def test_rule_based_calendar_events_are_marked_estimated():
    generated = events([])["events"]
    rule_events = [x for x in generated if x["category"] in ("derivatives", "commodities")]
    assert rule_events
    assert all(x.get("estimated") is True for x in rule_events)


def test_btc_metrics_payload_reuses_warm_cache():
    btc_onchain_metrics.clear_cache()
    with patch.object(btc_onchain_metrics, "_build_payload", return_value={"ok": True}) as build:
        assert btc_onchain_metrics.payload() == {"ok": True}
        assert btc_onchain_metrics.payload() == {"ok": True}
        assert build.call_count == 1
    btc_onchain_metrics.clear_cache()
