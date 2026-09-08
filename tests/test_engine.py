import numpy as np
import pandas as pd

from engine.backtesting import Config, _bar_index, brt_hour, run_backtest
from engine.optimizer import _is_valid
from engine.regime_detection import _fit_hmm


def _ohlc(n=400, tz=None, unit=None):
    close = pd.Series(np.abs(np.cumsum(np.random.default_rng(1).normal(0, 30, n)) + 40000))
    idx = pd.date_range("2021-06-01", periods=n, freq="h", tz=tz)
    if unit:
        idx = idx.as_unit(unit)
    return pd.DataFrame({"Open": close.values, "High": (close * 1.01).values,
                         "Low": (close * 0.99).values, "Close": close.values,
                         "Volume": np.ones(n)}, index=idx)


def test_bar_index_timestamps_are_milliseconds_for_every_index_unit():
    # asi8 segue a unidade do índice (us/ms/ns): sem as_unit("ns") o epoch sai
    # dividido por 1000 e o funding é atribuído à barra errada.
    for unit in ("ns", "us", "ms", "s"):
        idx = pd.date_range("2021-06-01", periods=5, freq="h").as_unit(unit)
        _, _, _, ts_ms = _bar_index(idx)
        assert ts_ms[0] == idx[0].value // 1_000_000, unit


def test_bar_index_matches_brt_hour_naive_and_tz_aware():
    for tz in (None, "UTC", "America/New_York"):
        idx = pd.date_range("2021-06-01", periods=48, freq="h", tz=tz)
        _, _, hours, _ = _bar_index(idx)
        assert hours == [brt_hour(t) for t in idx], tz


def test_bar_index_tolerates_non_datetime_index():
    dates, months, hours, ts = _bar_index(pd.RangeIndex(3))
    assert dates == ["0", "1", "2"] and months == [1, 1, 1]
    assert hours == [0, 0, 0] and ts == [0, 0, 0]


def test_backtest_is_stable_across_index_units_and_timezones():
    base = run_backtest(_ohlc(), Config())
    for kwargs in ({"unit": "ms"}, {"unit": "ns"}, {"tz": "UTC"}):
        st = run_backtest(_ohlc(**kwargs), Config())
        assert len(st.trades) == len(base.trades), kwargs
        assert st.equity == base.equity, kwargs


def test_invalid_configs_are_the_ones_varying_an_inactive_parameter():
    assert _is_valid({"use_stop": True, "stop_type": "ATR", "stop_atr_mult": 3.0})
    assert not _is_valid({"use_stop": True, "stop_type": "ATR", "stop_fixo_pct": 3.0})
    assert not _is_valid({"use_stop": False, "stop_atr_mult": 3.0})
    assert not _is_valid({"use_stop": False, "stop_type": "Banda Stop"})
    assert _is_valid({"exit_mode": "Alvo Fixo + Tendência", "alvo_fixo": 8.0})
    assert not _is_valid({"exit_mode": "Alvo Fixo + Tendência", "pct_up": 8.0})
    assert not _is_valid({"exit_mode": "Somente Tendência", "pct_up": 8.0})


def test_hmm_separates_two_shifted_regimes():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (300, 2))
    X[150:] += 4.0
    fit = _fit_hmm(X, 2)

    assert set(np.unique(fit["states"])) <= {0, 1}
    assert np.allclose(fit["posteriors"].sum(axis=1), 1.0)
    assert np.allclose(fit["transmat"].sum(axis=1), 1.0)
    # cada metade cai majoritariamente num estado, e não no mesmo
    first, second = np.bincount(fit["states"][:150]).argmax(), np.bincount(fit["states"][150:]).argmax()
    assert first != second
