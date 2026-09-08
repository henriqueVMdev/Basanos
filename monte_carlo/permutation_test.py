"""
Permutation Test — Timothy Masters Method.

Duas variantes:
  - Web/API: usa a equity curve (sem re-executar a estratégia; rápido).
  - CLI standalone: usa os log-retornos dos preços OHLC e re-executa a
    estratégia em cada série sintética (teste completo, mais lento).
"""
from __future__ import annotations

import numpy as np


# Helpers
def _sharpe_from_curve(equity: np.ndarray, ann_factor: float = 252.0) -> float:
    arr  = np.asarray(equity, dtype=float)
    rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    return float(rets.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0


def _profit_factor_from_trades(trades: list[dict]) -> float:
    wins   = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0]
    losses = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) <= 0]
    denom  = abs(sum(losses))
    return round(sum(wins) / denom, 4) if denom > 0 else float("inf")


# Variante 1: Equity-curve based (para uso na API/web)
class PermutationTestEquity:
    """
    Teste de permutação usando a equity curve diária.

    Processo:
      1. Calcula os log-retornos da equity curve original.
      2. Para cada permutação: embaralha os log-retornos e reconstrói a curva.
      3. Compara o Sharpe original com a distribuição dos Sharpes permutados.
      4. p-value = proporção de permutações com Sharpe >= original.

    Vantagem: extremamente rápido (sem re-executar a estratégia).
    """

    def __init__(self, seed: int | None = None, ann_factor: float = 252.0):
        self.rng        = np.random.default_rng(seed)
        self.ann_factor = float(ann_factor)

    def run(
        self,
        equity_values: list[float],
        trades: list[dict],
        n_perms: int = 500,
    ) -> dict:
        arr      = np.array(equity_values, dtype=float)
        log_rets = np.diff(np.log(np.where(arr > 0, arr, 1e-10)))
        log_rets = log_rets[np.isfinite(log_rets)]

        original_sharpe = _sharpe_from_curve(arr, self.ann_factor)
        original_pf     = _profit_factor_from_trades(trades)

        sharpe_dist = []
        pf_dist     = []

        for _ in range(n_perms):
            shuffled  = self.rng.permutation(log_rets)
            # Reconstrói equity: equity[t] = equity[0] * exp(sum(log_rets[:t]))
            log_cumsums = np.concatenate([[0.0], np.cumsum(shuffled)])
            perm_equity = arr[0] * np.exp(log_cumsums)
            sharpe_dist.append(_sharpe_from_curve(perm_equity, self.ann_factor))
            pf_dist.append(float("nan"))  # PF não disponível sem trades sintéticos

        sharpe_arr = np.array(sharpe_dist)
        valid_s    = sharpe_arr[np.isfinite(sharpe_arr)]

        p_value         = float(np.mean(valid_s >= original_sharpe)) if len(valid_s) > 0 else 1.0
        backtest_rank   = float(np.mean(valid_s <= original_sharpe) * 100) if len(valid_s) > 0 else 50.0
        approved        = p_value < 0.05

        return {
            "p_value":         round(p_value, 4),
            "sharpe_dist":     sharpe_dist,
            "original_sharpe": round(original_sharpe, 4),
            "original_pf":     round(original_pf, 4) if np.isfinite(original_pf) else None,
            "median_random":   round(float(np.nanmedian(valid_s)), 4) if len(valid_s) > 0 else 0.0,
            "backtest_rank":   round(backtest_rank, 1),
            "conclusion":      "APROVADO" if approved else "REPROVADO",
            "approved":        approved,
            "n_perms":         n_perms,
        }


# Variante 2: Full OHLC-based (Timothy Masters — para uso no CLI)
class PermutationTest:
    """
    Teste de permutação completo (Timothy Masters).

    Processo:
      1. Recebe série OHLC original e a estratégia (callable).
      2. Calcula log-retornos do Close.
      3. Para cada permutação:
           a. Embaralha os log-retornos (destrói padrões temporais).
           b. Reconstrói série Close sintética.
           c. Escala Open/High/Low proporcionalmente ao Close.
           d. Executa a estratégia na série sintética.
           e. Registra Sharpe e Profit Factor.
      4. p-value = proporção de permutações com Sharpe >= original.
    """

    def __init__(self, seed: int | None = None, ann_factor: float = 252.0):
        self.rng        = np.random.default_rng(seed)
        self.ann_factor = float(ann_factor)

    def run(
        self,
        df_ohlc,           # pd.DataFrame com colunas Open, High, Low, Close
        strategy_fn,       # callable(df) -> dict com keys: metrics, trades, equity_curve
        n_perms: int = 1000,
        metric: str = "sharpe",   # 'sharpe' | 'profit_factor'
    ) -> dict:
        """
        Parameters
        ----------
        df_ohlc : pd.DataFrame
            OHLC com índice datetime.
        strategy_fn : callable
            Função que recebe um DataFrame OHLC e retorna dict com 'metrics'.
        n_perms : int
            Número de permutações.
        metric : str
            Métrica de comparação ('sharpe' ou 'profit_factor').
        """
        import pandas as pd

        closes    = df_ohlc["Close"].values.astype(float)
        log_rets  = np.diff(np.log(np.where(closes > 0, closes, 1e-10)))

        # Backtest original
        orig_result = strategy_fn(df_ohlc)
        orig_metrics = orig_result.get("metrics", {})
        # Sharpe pode não vir direto; tenta calcular da equity curve
        orig_sharpe = orig_metrics.get("sharpe")
        if orig_sharpe is None:
            eq_vals = orig_result.get("equity_curve", {}).get("values", [])
            orig_sharpe = _sharpe_from_curve(np.array(eq_vals), self.ann_factor) if eq_vals else 0.0
        orig_pf = orig_metrics.get("profit_factor") or 1.0

        sharpe_dist = []
        pf_dist     = []
        print(f"[PermutationTest] Rodando {n_perms} permutações...")

        for i in range(n_perms):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_perms}")

            shuffled     = self.rng.permutation(log_rets)
            log_cumsums  = np.concatenate([[0.0], np.cumsum(shuffled)])
            synth_close  = closes[0] * np.exp(log_cumsums)

            # Escala O/H/L proporcionalmente
            ratios     = synth_close / np.where(closes > 0, closes, 1.0)
            synth_open = df_ohlc["Open"].values  * ratios
            synth_high = df_ohlc["High"].values  * ratios
            synth_low  = df_ohlc["Low"].values   * ratios

            synth_df = pd.DataFrame(
                {"Open": synth_open, "High": synth_high,
                 "Low": synth_low,  "Close": synth_close},
                index=df_ohlc.index,
            )
            try:
                res      = strategy_fn(synth_df)
                met      = res.get("metrics", {})
                s        = met.get("sharpe")
                if s is None:
                    eq_v = res.get("equity_curve", {}).get("values", [])
                    s    = _sharpe_from_curve(np.array(eq_v), self.ann_factor) if eq_v else 0.0
                pf       = met.get("profit_factor") or 1.0
                sharpe_dist.append(float(s))
                pf_dist.append(float(pf) if np.isfinite(float(pf)) else 1.0)
            except Exception:
                sharpe_dist.append(0.0)
                pf_dist.append(1.0)

        sharpe_arr  = np.array(sharpe_dist)
        pf_arr      = np.array(pf_dist)

        if metric == "profit_factor":
            dist_arr     = pf_arr
            orig_val     = float(orig_pf) if np.isfinite(float(orig_pf)) else 1.0
        else:
            dist_arr     = sharpe_arr
            orig_val     = float(orig_sharpe)

        p_value       = float(np.mean(dist_arr >= orig_val))
        backtest_rank = float(np.mean(dist_arr <= orig_val) * 100)
        approved      = p_value < 0.05

        return {
            "p_value":         round(p_value, 4),
            "sharpe_dist":     sharpe_dist,
            "pf_dist":         pf_dist,
            "original_sharpe": round(float(orig_sharpe), 4),
            "original_pf":     round(float(orig_pf), 4) if np.isfinite(float(orig_pf)) else None,
            "median_random":   round(float(np.median(sharpe_arr)), 4),
            "backtest_rank":   round(backtest_rank, 1),
            "conclusion":      "APROVADO" if approved else "REPROVADO",
            "approved":        approved,
            "n_perms":         n_perms,
        }
