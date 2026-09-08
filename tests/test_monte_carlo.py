import numpy as np

from monte_carlo.monte_carlo import MonteCarlo, _equity_from_trades, _max_drawdown_pct

TRADES = [{"pnl_pct": float(x)}
          for x in np.random.default_rng(4).normal(0.3, 2.5, 120)]


def test_equity_curve_compounds_from_the_initial_capital():
    eq = _equity_from_trades([10.0, -50.0, 100.0], 1000.0)
    assert eq.tolist() == [1000.0, 1100.0, 550.0, 1100.0]


def test_max_drawdown_is_negative_and_zero_when_monotonic():
    assert _max_drawdown_pct(np.array([100.0, 200.0, 300.0])) == 0.0
    assert _max_drawdown_pct(np.array([100.0, 50.0])) == -50.0
    assert _max_drawdown_pct(np.array([100.0])) == 0.0


def test_reshuffling_trades_cannot_change_the_final_equity():
    """Permutar os mesmos trades dá o mesmo produto: o rank tem que ser 50%,
    não o ruído de arredondamento da ordem da multiplicação."""
    res = MonteCarlo(10_000.0, seed=7).reshuffle(TRADES, n_sims=200)
    finals = np.array(res["final_equity_dist"])
    assert finals.max() - finals.min() < 1e-6      # só ruído de float
    assert res["backtest_rank_equity"] == 50.0


def test_shuffling_daily_returns_keeps_equity_and_sharpe_but_moves_drawdown():
    eq = list(10_000 * np.cumprod(1 + np.random.default_rng(4).normal(0.001, 0.02, 250)))
    res = MonteCarlo(10_000.0, seed=7).return_alteration(eq, list(range(250)), n_sims=200)
    assert res["backtest_rank_equity"] == 50.0
    assert res["backtest_rank_sharpe"] == 50.0
    dd = np.array(res["max_drawdown_dist"])
    assert dd.max() - dd.min() > 1.0               # o caminho muda o drawdown


def test_resample_produces_a_real_distribution():
    res = MonteCarlo(10_000.0, seed=7).resample(TRADES, n_sims=200)
    finals = np.array(res["final_equity_dist"])
    assert finals.max() - finals.min() > 1.0
    assert 0.0 <= res["backtest_rank_equity"] <= 100.0
