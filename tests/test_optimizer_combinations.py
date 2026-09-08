import pandas as pd

import server
from strategies import depaula


def test_depaula_presets_are_counted_without_cartesian_expansion():
    assert depaula.count_optimizer_configs(depaula.OPTIMIZER_GRIDS["rapido"]) == 20_736
    assert depaula.count_optimizer_configs(depaula.OPTIMIZER_GRIDS["custom"]) == 2_407_680
    assert depaula.count_optimizer_configs(depaula.OPTIMIZER_GRIDS["completo"]) == 608_256_000


def test_inactive_depaula_parameters_are_canonicalized_instead_of_rejected():
    cases = [
        ({"use_stop": [False], "stop_type": ["Banda Stop"],
          "stop_band_pct": [1.0, 2.0]}, "stop_type", "ATR"),
        ({"exit_mode": ["Somente Tendência"], "pct_up": [1.0, 2.0, 4.0]},
         "pct_up", 3.0),
        ({"use_parcial": [False], "parcial_pct": [25.0, 75.0]},
         "parcial_pct", 50.0),
    ]

    for grid, key, expected in cases:
        assert depaula.count_optimizer_configs(grid) == 1
        configs = list(depaula.iter_optimizer_configs(grid))
        assert len(configs) == 1
        assert configs[0][key] == expected
        assert depaula.is_valid_config(configs[0])


def test_stop_grid_only_branches_for_the_active_stop_type():
    grid = {
        "use_stop": [False, True],
        "stop_type": ["ATR", "Fixo (%)", "Banda Stop"],
        "stop_atr_mult": [1.0, 2.0, 3.0],
        "stop_fixo_pct": [1.0, 2.0, 3.0],
        "stop_band_pct": [1.0, 2.0, 3.0],
    }
    configs = list(depaula.iter_optimizer_configs(grid))
    assert depaula.count_optimizer_configs(grid) == 10
    assert len(configs) == 10
    assert all(depaula.is_valid_config(params) for params in configs)


def test_optimizer_count_endpoint_reports_large_grid_instead_of_timing_out():
    client = server.app.test_client()
    response = client.post("/api/optimizer/count", json={
        "strategy_file": "depaula",
        "grid": depaula.OPTIMIZER_GRIDS["completo"],
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["count"] == 608_256_000
    assert payload["exact"] is True
    assert payload["too_many"] is True
    assert payload["max_combinations"] == server._OPTIMIZER_MAX_COMBINATIONS


def test_empty_grid_value_is_an_explicit_zero_count():
    client = server.app.test_client()
    response = client.post("/api/optimizer/count", json={
        "strategy_file": "depaula",
        "grid": {"ma_type": []},
    })
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["count"] == 0
    assert payload["message"]


def test_optimizer_run_rejects_oversized_grid_before_touching_market_data():
    data, error = server._run_optimizer_compute(
        pd.DataFrame(),
        depaula,
        depaula.OPTIMIZER_GRIDS["completo"],
        1_000.0,
        5,
        "Score",
        20,
        "BTC",
        "1d",
    )
    assert data is None
    assert "Grid muito grande" in error

