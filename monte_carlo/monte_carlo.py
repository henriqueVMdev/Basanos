"""
Monte Carlo Simulation — 4 métodos para validação de estratégias.

Métodos disponíveis:
  reshuffle         — embaralha a ordem dos trades sem reposição
  resample          — bootstrap com reposição
  randomized        — adiciona ruído gaussiano ao PnL de cada trade
  return_alteration — embaralha os retornos diários da equity curve
"""
from __future__ import annotations

import numpy as np


# Helpers internos
def _equity_from_trades(pnls: np.ndarray, initial_capital: float) -> np.ndarray:
    """Constrói equity curve a partir de lista de PnL% por trade."""
    growth = np.cumprod(1.0 + np.asarray(pnls, dtype=float) / 100.0)
    return initial_capital * np.concatenate(([1.0], growth))


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Max drawdown em % (valor negativo)."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return float(min(0.0, ((eq - peak) / peak * 100.0).min()))


def _sharpe_from_pnls(pnls: np.ndarray) -> float:
    """Sharpe por trade — usado para comparação interna entre simulações MC."""
    arr = np.asarray(pnls, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    return float(arr.mean() / std * np.sqrt(len(arr))) if std > 0 else 0.0


def _sharpe_from_curve(equity: np.ndarray, ann_factor: float = 252.0) -> float:
    """Sharpe anualizado a partir da equity curve. ann_factor depende do timeframe."""
    arr  = np.asarray(equity, dtype=float)
    rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    return float(rets.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0


def _percentile_dict(arr: np.ndarray) -> dict:
    return {
        "p5":  round(float(np.percentile(arr, 5)),  4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def _build_result(
    method: str,
    curves: list[np.ndarray],
    pnls_list: list[np.ndarray] | None,
    original_equity: np.ndarray,
    original_sharpe: float,
    initial_capital: float,
    rng: np.random.Generator,
    ann_factor: float = 252.0,
) -> dict:
    """Agrega os resultados de N simulações num dict serializável."""
    curves_arr = np.array(curves)               # (n_sims, n_steps)
    n_sims = len(curves_arr)

    finals_abs = curves_arr[:, -1]
    finals_pct = (finals_abs / initial_capital - 1.0) * 100.0

    dds = np.array([_max_drawdown_pct(c) for c in curves_arr])

    if pnls_list is not None:
        sharpes = np.array([_sharpe_from_pnls(p) for p in pnls_list])
    else:
        sharpes = np.array([_sharpe_from_curve(c, ann_factor) for c in curves_arr])

    # Bandas percentis ao longo do tempo
    p5  = np.percentile(curves_arr, 5,  axis=0).tolist()
    p25 = np.percentile(curves_arr, 25, axis=0).tolist()
    p50 = np.percentile(curves_arr, 50, axis=0).tolist()
    p75 = np.percentile(curves_arr, 75, axis=0).tolist()
    p95 = np.percentile(curves_arr, 95, axis=0).tolist()

    # 100 caminhos amostrados para o broom chart
    n_sample = min(100, n_sims)
    idx = rng.choice(n_sims, size=n_sample, replace=False)
    sample_paths = [curves_arr[i].tolist() for i in idx]

    # Probabilidade de ruína: capital final < 50% do inicial
    ruin_prob = float(np.mean(finals_abs < initial_capital * 0.5) * 100.0)

    # Ranks percentis do backtest original vs distribuição simulada
    # Método "mean": (n_abaixo + 0.5 * n_igual) / n, para não reportar 100%
    # quando os valores são todos iguais (reshuffle não muda o equity final).
    orig_final_pct = float((original_equity[-1] / initial_capital - 1.0) * 100.0)
    orig_dd = float(_max_drawdown_pct(original_equity))

    def _rank_mean(dist: np.ndarray, value: float) -> float:
        n = len(dist)
        if n == 0:
            return 0.0
        # Tolerância: permutar os mesmos trades dá o mesmo equity a menos dos
        # últimos bits (multiplicação não é associativa). Sem isso, o rank do
        # reshuffle vira ruído de arredondamento em vez de 50%.
        tol = 1e-9 * max(1.0, abs(value))
        n_below = float(np.sum(dist < value - tol))
        n_equal = float(np.sum(np.abs(dist - value) <= tol))
        return (n_below + 0.5 * n_equal) / n * 100.0

    rank_equity = _rank_mean(finals_pct, orig_final_pct)
    rank_sharpe = _rank_mean(sharpes,    original_sharpe)
    rank_dd     = _rank_mean(-dds,       -orig_dd)  # dd mais baixo = melhor (inverte sinal)

    return {
        "method": method,
        # Broom chart
        "sample_paths":    sample_paths,
        "original_equity": original_equity.tolist(),
        "p5":  p5,  "p25": p25, "p50": p50, "p75": p75, "p95": p95,
        # Distribuições para histogramas
        "max_drawdown_dist":   dds.tolist(),
        "sharpe_dist":         sharpes.tolist(),
        "final_equity_dist":   finals_pct.tolist(),
        # Percentis das métricas
        "final_equity_pct": _percentile_dict(finals_pct),
        "max_drawdown":     _percentile_dict(dds),
        "sharpe":           _percentile_dict(sharpes),
        # Sumário
        "ruin_prob":            round(ruin_prob, 2),
        "backtest_rank_equity": round(rank_equity, 1),
        "backtest_rank_sharpe": round(rank_sharpe, 1),
        "backtest_rank_dd":     round(rank_dd, 1),
        "n_sims":               n_sims,
        "original_final_pct":   round(orig_final_pct, 2),
        "original_dd":          round(orig_dd, 2),
        "original_sharpe":      round(original_sharpe, 4),
    }


# Classe principal
class MonteCarlo:
    """
    Monte Carlo para validação de estratégias de trading.

    Parameters
    ----------
    initial_capital : float
        Capital inicial do backtest (ex.: 10 000).
    seed : int | None
        Semente para reprodutibilidade.
    """

    def __init__(self, initial_capital: float = 10_000.0, seed: int | None = None, ann_factor: float = 252.0):
        self.ic         = float(initial_capital)
        self.rng        = np.random.default_rng(seed)
        self.ann_factor = float(ann_factor)

    # Método 1: Reshuffle
    def reshuffle(self, trades: list[dict], n_sims: int = 1000) -> dict:
        """
        Embaralha a ordem dos trades (sem reposição).
        Todas as equity curves terminam no mesmo PnL total.
        """
        pnls = np.array([t["pnl_pct"] for t in trades], dtype=float)
        n = len(pnls)

        curves, pnls_list = [], []
        for _ in range(n_sims):
            shuffled = self.rng.permutation(pnls)
            curves.append(_equity_from_trades(shuffled, self.ic))
            pnls_list.append(shuffled)

        original = _equity_from_trades(pnls, self.ic)
        result = _build_result(
            "reshuffle", curves, pnls_list, original,
            _sharpe_from_pnls(pnls), self.ic, self.rng, self.ann_factor,
        )
        result["x_labels"] = list(range(n + 1))
        result["x_type"] = "trade_index"
        return result

    # Método 2: Resample
    def resample(self, trades: list[dict], n_sims: int = 1000) -> dict:
        """
        Bootstrap com reposição.
        Equity curves NÃO terminam no mesmo valor.
        """
        pnls = np.array([t["pnl_pct"] for t in trades], dtype=float)
        n = len(pnls)

        curves, pnls_list = [], []
        for _ in range(n_sims):
            sampled = self.rng.choice(pnls, size=n, replace=True)
            curves.append(_equity_from_trades(sampled, self.ic))
            pnls_list.append(sampled)

        original = _equity_from_trades(pnls, self.ic)
        result = _build_result(
            "resample", curves, pnls_list, original,
            _sharpe_from_pnls(pnls), self.ic, self.rng, self.ann_factor,
        )
        result["x_labels"] = list(range(n + 1))
        result["x_type"] = "trade_index"
        return result

    # Método 3: Randomized
    def randomized(
        self,
        trades: list[dict],
        n_sims: int = 1000,
        noise_pct: float = 0.1,
    ) -> dict:
        """
        Adiciona ruído gaussiano de ±(noise_pct × desvio_padrão) ao PnL de cada trade.
        Testa robustez quando resultados individuais variam levemente.
        """
        pnls = np.array([t["pnl_pct"] for t in trades], dtype=float)
        n = len(pnls)
        sigma_noise = float(pnls.std() * noise_pct)

        curves, pnls_list = [], []
        for _ in range(n_sims):
            noise  = self.rng.normal(0.0, sigma_noise, size=n)
            noisy  = pnls + noise
            curves.append(_equity_from_trades(noisy, self.ic))
            pnls_list.append(noisy)

        original = _equity_from_trades(pnls, self.ic)
        result = _build_result(
            "randomized", curves, pnls_list, original,
            _sharpe_from_pnls(pnls), self.ic, self.rng, self.ann_factor,
        )
        result["x_labels"] = list(range(n + 1))
        result["x_type"]   = "trade_index"
        result["noise_pct"] = noise_pct
        return result

    # Método 4: Return Alteration
    def return_alteration(
        self,
        equity_values: list[float],
        dates: list[str],
        n_sims: int = 1000,
    ) -> dict:
        """
        Embaralha os retornos percentuais diários da equity curve (período a período).
        Trabalha com a equity curve diária, não com trades individuais.
        """
        arr  = np.array(equity_values, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        n    = len(rets)

        curves = []
        for _ in range(n_sims):
            shuffled = self.rng.permutation(rets)
            equity   = np.empty(n + 1)
            equity[0] = arr[0]
            for i, r in enumerate(shuffled):
                equity[i + 1] = equity[i] * (1.0 + r)
            curves.append(equity)

        result = _build_result(
            "return_alteration", curves, None, arr,
            _sharpe_from_curve(arr, self.ann_factor), self.ic, self.rng, self.ann_factor,
        )
        result["x_labels"] = dates  # strings de data
        result["x_type"]   = "date"
        return result
