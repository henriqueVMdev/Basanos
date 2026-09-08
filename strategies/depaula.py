"""
Estratégia DePaula — Ângulo de Média Móvel
==========================================
Baseada no ângulo (derivada normalizada) de uma média móvel para detectar tendência.
Suporta entradas por pullback, zonas de entrada, stop loss configurável e múltiplos
modos de saída.

Usa o motor em backtesting.py (run_backtest + Config).
"""

import sys
import os
import math
import itertools
import numpy as np

# Garante que o diretório raiz do projeto está no sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.backtesting import run_backtest, Config

NAME = "DePaula — Ângulo de MA"
DESCRIPTION = "Detecta tendências pelo ângulo da média móvel. Suporta pullback, stop e múltiplos modos de saída."

CONFIG_SCHEMA = [
    {
        "title": "Média Móvel",
        "fields": [
            {
                "key": "ma_type",
                "label": "Tipo",
                "type": "select",
                "default": "HMA",
                "options": ["SMA", "EMA", "HMA", "RMA", "WMA"],
            },
            {
                "key": "ma_length",
                "label": "Período",
                "type": "number",
                "default": 50,
                "min": 2,
                "max": 500,
                "step": 1,
            },
            {
                "key": "lookback",
                "label": "Lookback",
                "type": "number",
                "default": 5,
                "min": 2,
                "max": 100,
                "step": 1,
            },
            {
                "key": "th_up",
                "label": "Ângulo Alta",
                "type": "number",
                "default": 0.5,
                "step": 0.1,
            },
            {
                "key": "th_dn",
                "label": "Ângulo Baixa",
                "type": "number",
                "default": -0.5,
                "step": 0.1,
                "optimizer_hidden": True,
            },
            {
                "key": "hysteresis",
                "label": "Histerese",
                "type": "number",
                "default": 0.2,
                "step": 0.1,
            },
        ],
    },
    {
        "title": "Saída",
        "fields": [
            {
                "key": "exit_mode",
                "label": "Modo de Saída",
                "type": "select",
                "default": "Banda + Tendência",
                "options": [
                    "Banda + Tendência",
                    "Somente Tendência",
                    "Alvo Fixo + Tendência",
                ],
            },
            {
                "key": "pct_up",
                "label": "Banda Sup (%)",
                "type": "number",
                "default": 3.0,
                "step": 0.5,
            },
            {
                "key": "pct_dn",
                "label": "Banda Inf (%)",
                "type": "number",
                "default": 3.0,
                "step": 0.5,
                "optimizer_hidden": True,
            },
            {
                "key": "alvo_fixo",
                "label": "Alvo Fixo (%)",
                "type": "number",
                "default": 5.0,
                "step": 0.5,
            },
            {
                "key": "exit_on_flat",
                "label": "Sair no Flat (cinza)",
                "type": "checkbox",
                "default": True,
            },
        ],
    },
    {
        "title": "Saida Parcial",
        "fields": [
            {
                "key": "use_parcial",
                "label": "Usar Parcial",
                "type": "checkbox",
                "default": False,
            },
            {
                "key": "parcial_pct",
                "label": "Realizar (%)",
                "type": "number",
                "default": 50.0,
                "min": 1,
                "max": 99,
                "step": 5,
                "show_if": "use_parcial",
            },
            {
                "key": "parcial_mode",
                "label": "Alvo Parcial",
                "type": "select",
                "default": "Banda",
                "options": ["Banda", "Alvo Fixo"],
                "show_if": "use_parcial",
            },
            {
                "key": "parcial_banda_pct",
                "label": "Banda Parcial (%)",
                "type": "number",
                "default": 1.5,
                "step": 0.5,
                "show_if": "use_parcial",
            },
            {
                "key": "parcial_alvo_fixo",
                "label": "Alvo Fixo Parcial (%)",
                "type": "number",
                "default": 2.0,
                "step": 0.5,
                "show_if": "use_parcial",
            },
        ],
    },
    {
        "title": "Stop Loss",
        "fields": [
            {
                "key": "use_stop",
                "label": "Usar Stop",
                "type": "checkbox",
                "default": False,
            },
            {
                "key": "stop_type",
                "label": "Tipo de Stop",
                "type": "select",
                "default": "ATR",
                "options": ["ATR", "Fixo (%)", "Banda Stop"],
                "show_if": "use_stop",
            },
            {
                "key": "stop_atr_mult",
                "label": "ATR Mult",
                "type": "number",
                "default": 2.0,
                "step": 0.1,
                "show_if": "use_stop",
            },
            {
                "key": "stop_fixo_pct",
                "label": "Fixo (%)",
                "type": "number",
                "default": 2.0,
                "step": 0.5,
                "show_if": "use_stop",
            },
            {
                "key": "stop_band_pct",
                "label": "Banda (%)",
                "type": "number",
                "default": 1.5,
                "step": 0.5,
                "show_if": "use_stop",
            },
        ],
    },
    {
        "title": "Alavancagem / Sizing",
        "fields": [
            {
                "key": "sizing_mode",
                "label": "Modo de Sizing",
                "type": "select",
                "default": "Alavancagem fixa",
                "options": ["Alavancagem fixa", "Quantidade fixa", "Risco por trade"],
            },
            {
                "key": "leverage",
                "label": "Alavancagem (x)",
                "type": "number",
                "default": 1.0,
                "min": 1.0,
                "max": 125.0,
                "step": 1.0,
            },
            {
                "key": "margin_pct",
                "label": "Margem por Trade (% do capital)",
                "type": "number",
                "default": 100.0,
                "min": 1.0,
                "max": 100.0,
                "step": 5.0,
            },
            {
                "key": "fixed_qty",
                "label": "Quantidade Fixa (unidades)",
                "type": "number",
                "default": 0.1,
                "min": 0.0,
                "step": 0.01,
            },
            {
                "key": "risk_per_trade",
                "label": "Risco por Trade (%)",
                "type": "number",
                "default": 1.0,
                "min": 0.1,
                "max": 100,
                "step": 0.1,
            },
        ],
    },
    {
        "title": "Entrada",
        "fields": [
            {
                "key": "use_pullback",
                "label": "Pullback",
                "type": "checkbox",
                "default": True,
            },
            {
                "key": "use_entry_zone",
                "label": "Entry Zone",
                "type": "checkbox",
                "default": False,
            },
        ],
    },
]


# Grids de otimizacao (parametros e valores a testar)
OPTIMIZER_GRIDS = {
    "rapido": {
        "ma_type":       ["HMA", "EMA"],
        "ma_length":     [21, 50, 100],
        "lookback":      [3, 5, 8],
        "th_up":         [0.3, 0.5, 1.0],
        "exit_mode":     ["Banda + Tendência", "Somente Tendência"],
        "pct_up":        [2.0, 3.0, 5.0],
        "exit_on_flat":  [True, False],
        "use_stop":      [False, True],
        "stop_type":     ["ATR"],
        "stop_atr_mult": [1.5, 2.0, 3.0],
        "use_pullback":  [True, False],
        "use_norm":      [True, False],
        "atr_length":    [14],
        "hysteresis":    [0.1, 0.2, 0.5],
        "use_parcial":   [False],
        "cycle_filter":  [False],
    },
    "completo": {
        "ma_type":       ["SMA", "EMA", "HMA", "WMA"],
        "ma_length":     [14, 21, 34, 50, 100, 200],
        "lookback":      [3, 5, 8, 13],
        "th_up":         [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        "exit_mode":     ["Banda + Tendência", "Somente Tendência", "Alvo Fixo + Tendência"],
        "pct_up":        [1.5, 2.0, 3.0, 5.0, 8.0],
        "alvo_fixo":     [3.0, 5.0, 8.0, 10.0],
        "exit_on_flat":  [True, False],
        "use_stop":      [False, True],
        "stop_type":     ["ATR", "Fixo (%)", "Banda Stop"],
        "stop_atr_mult": [1.0, 1.5, 2.0, 3.0],
        "stop_fixo_pct": [1.0, 2.0, 3.0],
        "stop_band_pct": [1.0, 1.5, 2.0],
        "use_pullback":  [True, False],
        "use_entry_zone": [False, True],
        "use_norm":      [True, False],
        "atr_length":    [10, 14, 21],
        "hysteresis":    [0.0, 0.1, 0.2, 0.5],
        "use_parcial":       [False, True],
        "parcial_pct":       [25, 50, 75],
        "parcial_mode":      ["Banda", "Alvo Fixo"],
        "parcial_banda_pct": [1.0, 1.5, 2.0, 3.0],
        "parcial_alvo_fixo": [1.0, 2.0, 3.0, 5.0],
        "cycle_filter":      [False, True],
    },
    "custom": {
        "ma_type":       ["EMA", "SMA", "HMA"],
        "ma_length":     list(range(40, 150, 5)),
        "lookback":      [0, 5],
        "th_up":         [0.1, 0.5, 1.5, 2.0, 2.5],
        "exit_mode":     ["Banda + Tendência", "Somente Tendência"],
        "pct_up":        list(range(5, 40, 2)),
        "exit_on_flat":  [False, True],
        "use_stop":      [True],
        "stop_type":     ["Banda Stop"],
        "stop_band_pct": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "use_pullback":  [True, False],
        "use_entry_zone": [True],
        "use_norm":      [False],
        "atr_length":    [14],
        "hysteresis":    [0.2],
        "use_parcial":       [False, True],
        "parcial_pct":       [50],
        "parcial_mode":      ["Banda"],
        "parcial_banda_pct": [1.0, 1.5, 2.0],
        "parcial_alvo_fixo": [2.0],
        "cycle_filter":      [False, True],
    },
}


def is_valid_config(params):
    """Remove combinacoes que nao fazem sentido para esta estrategia."""
    if not params.get("use_stop", False):
        if params.get("stop_type", "ATR") != "ATR":
            return False
        if params.get("stop_atr_mult", 2.0) != 2.0:
            return False
        if params.get("stop_fixo_pct", 2.0) != 2.0:
            return False
        if params.get("stop_band_pct", 1.5) != 1.5:
            return False

    stop_type = params.get("stop_type", "ATR")
    if params.get("use_stop", False):
        if stop_type != "ATR" and params.get("stop_atr_mult", 2.0) != 2.0:
            return False
        if stop_type != "Fixo (%)" and params.get("stop_fixo_pct", 2.0) != 2.0:
            return False
        if stop_type != "Banda Stop" and params.get("stop_band_pct", 1.5) != 1.5:
            return False

    exit_mode = params.get("exit_mode", "Banda + Tendência")
    if exit_mode == "Somente Tendência":
        if params.get("pct_up", 3.0) != 3.0:
            return False
        if params.get("alvo_fixo", 5.0) != 5.0:
            return False
    if exit_mode == "Banda + Tendência":
        if params.get("alvo_fixo", 5.0) != 5.0:
            return False
    if exit_mode == "Alvo Fixo + Tendência":
        if params.get("pct_up", 3.0) != 3.0:
            return False

    # Saida parcial: se desativada, params parciais devem estar no default
    if not params.get("use_parcial", False):
        if params.get("parcial_pct", 50.0) != 50.0:
            return False
        if params.get("parcial_mode", "Banda") != "Banda":
            return False
        if params.get("parcial_banda_pct", 1.5) != 1.5:
            return False
        if params.get("parcial_alvo_fixo", 2.0) != 2.0:
            return False
    else:
        parcial_mode = params.get("parcial_mode", "Banda")
        if parcial_mode == "Banda" and params.get("parcial_alvo_fixo", 2.0) != 2.0:
            return False
        if parcial_mode == "Alvo Fixo" and params.get("parcial_banda_pct", 1.5) != 1.5:
            return False

    return True


_OPTIMIZER_DEFAULTS = {
    "exit_mode": "Banda + Tendência",
    "pct_up": 3.0,
    "pct_dn": 3.0,
    "alvo_fixo": 5.0,
    "use_stop": False,
    "stop_type": "ATR",
    "stop_atr_mult": 2.0,
    "stop_fixo_pct": 2.0,
    "stop_band_pct": 1.5,
    "use_parcial": False,
    "parcial_pct": 50.0,
    "parcial_mode": "Banda",
    "parcial_banda_pct": 1.5,
    "parcial_alvo_fixo": 2.0,
}

_OPTIMIZER_CONDITIONAL_KEYS = {
    "exit_mode", "pct_up", "pct_dn", "alvo_fixo",
    "use_stop", "stop_type", "stop_atr_mult", "stop_fixo_pct",
    "stop_band_pct",
    "use_parcial", "parcial_pct", "parcial_mode",
    "parcial_banda_pct", "parcial_alvo_fixo",
}


def _optimizer_values(grid, fixed_params, key):
    """Valores do grid; na ausência, usa config fixa e depois o default."""
    if key in grid:
        return list(grid[key])
    return [fixed_params.get(key, _OPTIMIZER_DEFAULTS[key])]


def _optimizer_exit_variants(grid, fixed_params):
    variants = []
    for mode in _optimizer_values(grid, fixed_params, "exit_mode"):
        if mode == "Banda + Tendência":
            for pct in _optimizer_values(grid, fixed_params, "pct_up"):
                variants.append({
                    "exit_mode": mode,
                    "pct_up": pct,
                    "pct_dn": pct,
                    "alvo_fixo": _OPTIMIZER_DEFAULTS["alvo_fixo"],
                })
        elif mode == "Alvo Fixo + Tendência":
            for alvo in _optimizer_values(grid, fixed_params, "alvo_fixo"):
                variants.append({
                    "exit_mode": mode,
                    "pct_up": _OPTIMIZER_DEFAULTS["pct_up"],
                    "pct_dn": _OPTIMIZER_DEFAULTS["pct_dn"],
                    "alvo_fixo": alvo,
                })
        else:  # Somente Tendência
            variants.append({
                "exit_mode": mode,
                "pct_up": _OPTIMIZER_DEFAULTS["pct_up"],
                "pct_dn": _OPTIMIZER_DEFAULTS["pct_dn"],
                "alvo_fixo": _OPTIMIZER_DEFAULTS["alvo_fixo"],
            })
    return variants


def _optimizer_stop_variants(grid, fixed_params):
    variants = []
    for enabled in _optimizer_values(grid, fixed_params, "use_stop"):
        if not enabled:
            variants.append({
                "use_stop": False,
                "stop_type": _OPTIMIZER_DEFAULTS["stop_type"],
                "stop_atr_mult": _OPTIMIZER_DEFAULTS["stop_atr_mult"],
                "stop_fixo_pct": _OPTIMIZER_DEFAULTS["stop_fixo_pct"],
                "stop_band_pct": _OPTIMIZER_DEFAULTS["stop_band_pct"],
            })
            continue

        for stop_type in _optimizer_values(grid, fixed_params, "stop_type"):
            base = {
                "use_stop": True,
                "stop_type": stop_type,
                "stop_atr_mult": _OPTIMIZER_DEFAULTS["stop_atr_mult"],
                "stop_fixo_pct": _OPTIMIZER_DEFAULTS["stop_fixo_pct"],
                "stop_band_pct": _OPTIMIZER_DEFAULTS["stop_band_pct"],
            }
            if stop_type == "ATR":
                key = "stop_atr_mult"
            elif stop_type == "Fixo (%)":
                key = "stop_fixo_pct"
            else:
                key = "stop_band_pct"
            for value in _optimizer_values(grid, fixed_params, key):
                variants.append({**base, key: value})
    return variants


def _optimizer_partial_variants(grid, fixed_params):
    variants = []
    for enabled in _optimizer_values(grid, fixed_params, "use_parcial"):
        if not enabled:
            variants.append({
                "use_parcial": False,
                "parcial_pct": _OPTIMIZER_DEFAULTS["parcial_pct"],
                "parcial_mode": _OPTIMIZER_DEFAULTS["parcial_mode"],
                "parcial_banda_pct": _OPTIMIZER_DEFAULTS["parcial_banda_pct"],
                "parcial_alvo_fixo": _OPTIMIZER_DEFAULTS["parcial_alvo_fixo"],
            })
            continue

        for fraction in _optimizer_values(grid, fixed_params, "parcial_pct"):
            for mode in _optimizer_values(grid, fixed_params, "parcial_mode"):
                base = {
                    "use_parcial": True,
                    "parcial_pct": fraction,
                    "parcial_mode": mode,
                    "parcial_banda_pct": _OPTIMIZER_DEFAULTS["parcial_banda_pct"],
                    "parcial_alvo_fixo": _OPTIMIZER_DEFAULTS["parcial_alvo_fixo"],
                }
                key = "parcial_banda_pct" if mode == "Banda" else "parcial_alvo_fixo"
                for value in _optimizer_values(grid, fixed_params, key):
                    variants.append({**base, key: value})
    return variants


def _optimizer_parts(grid, fixed_params=None):
    """Partes independentes/condicionais do grid, sem ramificar campos inativos."""
    fixed_params = dict(fixed_params or {})
    if any(not values for values in grid.values()):
        return [], [], [], [], []

    independent_keys = sorted(set(grid) - _OPTIMIZER_CONDITIONAL_KEYS)
    independent_values = [list(grid[key]) for key in independent_keys]
    return (
        independent_keys,
        independent_values,
        _optimizer_exit_variants(grid, fixed_params),
        _optimizer_stop_variants(grid, fixed_params),
        _optimizer_partial_variants(grid, fixed_params),
    )


def count_optimizer_configs(grid, fixed_params=None):
    """Conta o grid DePaula em tempo constante, ignorando campos inativos."""
    keys, values, exits, stops, partials = _optimizer_parts(grid, fixed_params)
    if not exits or not stops or not partials:
        return 0
    independent_count = math.prod(len(items) for items in values) if keys else 1
    return independent_count * len(exits) * len(stops) * len(partials)


def iter_optimizer_configs(grid, fixed_params=None):
    """Itera somente configurações semanticamente distintas e válidas."""
    keys, values, exits, stops, partials = _optimizer_parts(grid, fixed_params)
    if not exits or not stops or not partials:
        return

    independent_product = itertools.product(*values) if keys else [()]
    for raw_values in independent_product:
        independent = dict(zip(keys, raw_values))
        for exit_params, stop_params, partial_params in itertools.product(
                exits, stops, partials):
            params = {**independent, **exit_params, **stop_params, **partial_params}
            yield prepare_optimizer_params(params)


def prepare_optimizer_params(params):
    """Aplica regras de simetria antes de rodar."""
    if "th_up" in params:
        params["th_dn"] = -params["th_up"]
    if "pct_up" in params:
        params["pct_dn"] = params["pct_up"]
    return params


def _safe(val):
    """Converte NaN/Inf para None (JSON-safe)."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def run(df, params: dict) -> dict:
    """
    Executa o backtest DePaula.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV com colunas Open, High, Low, Close, Volume e índice datetime.
    params : dict
        Parâmetros do usuário (chaves = keys do CONFIG_SCHEMA).

    Returns
    -------
    dict com keys: metrics, equity_curve, trades
    """
    cfg = Config(
        ma_type=params.get("ma_type", "HMA"),
        ma_length=int(params.get("ma_length", 50)),
        lookback=int(params.get("lookback", 5)),
        th_up=float(params.get("th_up", 0.5)),
        th_dn=float(params.get("th_dn", -0.5)),
        hysteresis=float(params.get("hysteresis", 0.2)),
        exit_mode=params.get("exit_mode", "Banda + Tendência"),
        pct_up=float(params.get("pct_up", 3.0)),
        pct_dn=float(params.get("pct_dn", 3.0)),
        alvo_fixo=float(params.get("alvo_fixo", 5.0)),
        exit_on_flat=bool(params.get("exit_on_flat", True)),
        use_stop=bool(params.get("use_stop", False)),
        stop_type=params.get("stop_type", "ATR"),
        stop_atr_mult=float(params.get("stop_atr_mult", 2.0)),
        stop_fixo_pct=float(params.get("stop_fixo_pct", 2.0)),
        stop_band_pct=float(params.get("stop_band_pct", 1.5)),
        use_pullback=bool(params.get("use_pullback", True)),
        use_entry_zone=bool(params.get("use_entry_zone", False)),
        use_parcial=bool(params.get("use_parcial", False)),
        parcial_pct=float(params.get("parcial_pct", 50.0)),
        parcial_mode=params.get("parcial_mode", "Banda"),
        parcial_banda_pct=float(params.get("parcial_banda_pct", 1.5)),
        parcial_alvo_fixo=float(params.get("parcial_alvo_fixo", 2.0)),
        use_norm=bool(params.get("use_norm", True)),
        atr_length=int(params.get("atr_length", 14)),
        use_position_sizing=bool(params.get("use_position_sizing", False))
        or params.get("sizing_mode", "Alavancagem fixa") == "Risco por trade",
        risk_per_trade=float(params.get("risk_per_trade", 1.0)),
        sizing_mode=params.get("sizing_mode", "Alavancagem fixa"),
        leverage=float(params.get("leverage", 1.0)),
        margin_pct=float(params.get("margin_pct", 100.0)),
        fixed_qty=float(params.get("fixed_qty", 0.1)),
        initial_capital=float(params.get("initial_capital", 1000.0)),
        cycle_filter=bool(params.get("cycle_filter", False)),
        cycle_long_months=params.get("cycle_long_months", []),
        cycle_short_months=params.get("cycle_short_months", []),
        hour_filter=bool(params.get("hour_filter", False)),
        allowed_hours=params.get("allowed_hours", []),
    )

    result = run_backtest(df, cfg)
    bt_df = result._df
    trades = result.trades

    # Métricas
    total_return = (result.equity / cfg.initial_capital - 1) * 100
    wins = [t for t in trades if t.pnl_pct > 0]
    losses_t = [t for t in trades if t.pnl_pct <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0
    avg_loss = float(np.mean([t.pnl_pct for t in losses_t])) if losses_t else 0
    pf_num = sum(t.pnl_pct for t in wins)
    pf_den = abs(sum(t.pnl_pct for t in losses_t))
    profit_factor = pf_num / pf_den if pf_den > 0 else None

    eq = bt_df["Equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(dd.min())

    # Calmar — CAGR uses calendar days, not bar count
    total_days = max((bt_df.index[-1] - bt_df.index[0]).days, 1) if len(bt_df) > 1 else 1
    cagr = ((result.equity / cfg.initial_capital) ** (365.25 / total_days) - 1) * 100
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0

    # Annualization factor inferred from bar frequency
    bars_per_year = len(bt_df) / (total_days / 365.25) if total_days > 0 else 252

    # Sharpe from equity curve returns (annualized, ddof=1)
    sharpe = 0.0
    pnls = [t.pnl_pct for t in trades]
    if len(eq) > 2:
        eq_rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
        eq_rets = eq_rets[np.isfinite(eq_rets)]
        eq_std = float(eq_rets.std(ddof=1))
        if len(eq_rets) > 1 and eq_std > 0:
            sharpe = float(eq_rets.mean() / eq_std * np.sqrt(bars_per_year))

    # Sortino — annualized by trades_per_year, not total trades
    downside_sq = [min(p, 0) ** 2 for p in pnls]
    downside_dev = float(np.sqrt(np.mean(downside_sq))) if downside_sq else 0
    trades_per_year = len(pnls) / (total_days / 365.25) if total_days > 0 else len(pnls)
    sortino = float(np.mean(pnls) / downside_dev * np.sqrt(trades_per_year)) if downside_dev > 0 else 0

    # Omega (sum gains / sum |losses|)
    gains_sum = sum(p for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p < 0))
    omega = float(gains_sum / losses_sum) if losses_sum > 0 else None

    # Extract dd episode troughs for Sterling and Burke
    ep_troughs = []
    _in_dd = False
    _ep_start = None
    for _k, _val in enumerate(dd):
        if _val < 0 and not _in_dd:
            _in_dd = True
            _ep_start = _k
        elif _val >= 0 and _in_dd:
            _in_dd = False
            ep_troughs.append(float(dd[_ep_start:_k].min()))
    if _in_dd and _ep_start is not None:
        ep_troughs.append(float(dd[_ep_start:].min()))

    # Sterling (CAGR / mean of N worst episode troughs)
    if ep_troughs:
        n_worst = min(5, len(ep_troughs))
        worst_troughs = sorted(ep_troughs)[:n_worst]
        avg_worst_dd = abs(float(np.mean(worst_troughs)))
        sterling = float(cagr / avg_worst_dd) if avg_worst_dd > 0 else 0
    else:
        sterling = 0

    # Burke (CAGR / sqrt(sum of N worst episode trough^2))
    if ep_troughs:
        n_worst = min(5, len(ep_troughs))
        worst_troughs = sorted(ep_troughs)[:n_worst]
        burke_denom = float(np.sqrt(np.sum(np.array(worst_troughs) ** 2)))
        burke = float(cagr / burke_denom) if burke_denom > 0 else 0
    else:
        burke = 0

    # _fast=True: pula tudo que é só exibição (optimizer e amostragem IS do WFA)
    _fast = bool(params.get("_fast", False))

    # dates is needed by dd_episodes and equity_curve
    dates = [str(idx)[:10] for idx in bt_df.index]

    if _fast:
        recovery_factor = None
        ulcer_index     = None
        avg_dd          = None
        avg_dd_len      = None
        n_dd_episodes   = 0
        dd_dates        = []
        dd_values       = []
        equity_values   = [_safe(float(v)) for v in bt_df["Equity"].tolist()]
        trades_list     = [
            {
                "pnl_pct": _safe(float(t.pnl_pct)),
                "qty": _safe(float(t.qty)),
                "leverage": _safe(float(t.leverage)),
                "notional": _safe(float(t.notional)),
                "direction": t.direction,
                "entry_price": _safe(float(t.entry_price)),
                "exit_price": _safe(float(t.exit_price)),
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
            }
            for t in trades
        ]
    else:
        # Drawdown analytics
        net_profit = result.equity - cfg.initial_capital
        recovery_factor = _safe(float(net_profit / abs(max_dd * cfg.initial_capital / 100))) if max_dd < 0 else None

        ulcer_index = _safe(float(np.sqrt(np.mean(dd ** 2)))) if len(dd) > 0 else None

        dd_episodes = []
        in_dd = False
        ep_start = None
        ep_peak_idx = None
        for k, val in enumerate(dd):
            if val < 0 and not in_dd:
                in_dd = True
                ep_start = k
                ep_peak_idx = k
            elif val < 0 and in_dd:
                if val < dd[ep_peak_idx]:
                    ep_peak_idx = k
            elif val >= 0 and in_dd:
                in_dd = False
                ep_trough = float(dd[ep_start:k].min())
                ep_len    = k - ep_start
                dd_episodes.append({
                    "start": dates[ep_start] if ep_start < len(bt_df) else None,
                    "end":   dates[k - 1]   if (k - 1) < len(bt_df) else None,
                    "trough": _safe(ep_trough),
                    "length_bars": ep_len,
                })
        if in_dd and ep_start is not None:
            ep_trough = float(dd[ep_start:].min())
            dd_episodes.append({
                "start": dates[ep_start],
                "end":   dates[-1],
                "trough": _safe(ep_trough),
                "length_bars": len(dd) - ep_start,
            })

        avg_dd        = _safe(float(np.mean([e["trough"] for e in dd_episodes]))) if dd_episodes else None
        avg_dd_len    = _safe(float(np.mean([e["length_bars"] for e in dd_episodes]))) if dd_episodes else None
        n_dd_episodes = len(dd_episodes)
        dd_dates      = dates
        dd_values     = [_safe(float(v)) for v in dd.tolist()]

        equity_values = [_safe(float(v)) for v in bt_df["Equity"].tolist()]

        trades_list = [
            {
                "entry_date": t.entry_date[:10] if t.entry_date else "",
                "exit_date": t.exit_date[:10] if t.exit_date else "",
                "direction": t.direction,
                "comment": t.comment,
                "entry_price": _safe(float(t.entry_price)),
                "exit_price": _safe(float(t.exit_price)),
                "exit_comment": t.exit_comment,
                "pnl_pct": _safe(float(t.pnl_pct)),
                "partial_exit_price": _safe(float(t.partial_exit_price)) if t.partial_exit_price else None,
                "partial_exit_date": t.partial_exit_date[:10] if t.partial_exit_date else None,
                "partial_pct_closed": t.partial_pct_closed if t.partial_pct_closed else None,
                "qty": _safe(float(t.qty)),
                "leverage": _safe(float(t.leverage)),
                "notional": _safe(float(t.notional)),
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
            }
            for t in trades
        ]

    # Séries para os gráficos de análise (candles + indicadores)
    # Só montadas sob demanda (_charts), para não inchar a resposta padrão.
    chart = None
    if params.get("_charts"):
        def _col(name):
            return [_safe(float(v)) for v in bt_df[name].tolist()] if name in bt_df.columns else None
        chart = {
            "dates": [str(idx) for idx in bt_df.index],   # timestamp completo (intraday)
            "ohlc": {
                "open":   _col("Open"),
                "high":   _col("High"),
                "low":    _col("Low"),
                "close":  _col("Close"),
                "volume": _col("Volume"),
            },
            "indicators": {
                "ma":    _col("MA"),
                "upper": _col("UpperBand"),
                "lower": _col("LowerBand"),
            },
        }

    return {
        "chart": chart,
        "metrics": {
            "final_equity": _safe(float(result.equity)),
            "total_return": _safe(float(total_return)),
            "max_dd": _safe(max_dd),
            "total_trades": len(trades),
            "win_rate": _safe(float(win_rate)),
            "profit_factor": _safe(profit_factor),
            "avg_win": _safe(avg_win),
            "avg_loss": _safe(avg_loss),
            "sharpe": _safe(sharpe),
            "sortino": _safe(sortino),
            "calmar": _safe(calmar),
            "omega": _safe(omega),
            "sterling": _safe(sterling),
            "burke": _safe(burke),
            "recovery_factor": recovery_factor,
            "ulcer_index": ulcer_index,
            "avg_dd": avg_dd,
            "avg_dd_length": avg_dd_len,
            "n_dd_episodes": n_dd_episodes,
            "initial_capital": float(cfg.initial_capital),
        },
        "drawdown": {
            "dates":    dd_dates,
            "values":   dd_values,
            "episodes": dd_episodes if not _fast else [],
        },
        "equity_curve": {
            "dates": dates,
            "values": equity_values,
        },
        "trades": trades_list,
    }
