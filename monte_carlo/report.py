"""
Geração de relatório textual de validação da estratégia.
Pode ser impresso no terminal ou salvo em output/report.txt.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    return f"{n}{suffix}"


def _fmt_money(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "-"
    return f"${v:,.2f}"


def _fmt_pct(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "-"
    return f"{v:+.2f}%"


def _check(condition: bool) -> str:
    return "[✓]" if condition else "[✗]"


def generate(
    original: dict,
    mc_results: dict[str, dict],
    perm_result: dict,
    strategy_label: str = "Estratégia",
    asset_label: str = "-",
    period_label: str = "-",
    n_sims: int = 1000,
) -> str:
    """
    Gera o relatório completo como string.

    Parameters
    ----------
    original : dict
        Métricas originais: initial_capital, final_equity, total_return,
        sharpe, max_dd, win_rate, profit_factor, total_trades,
        avg_win, avg_loss, expectancy (opcional).
    mc_results : dict[str, dict]
        { 'reshuffle': SimResult, 'resample': ..., 'randomized': ..., 'return_alteration': ... }
        Usa o 'reshuffle' como referência principal.
    perm_result : dict
        Resultado do PermutationTest.
    """
    ic         = float(original.get("initial_capital", 10_000))
    final_eq   = float(original.get("final_equity", ic))
    total_ret  = float(original.get("total_return", 0))
    sharpe     = original.get("sharpe") or 0.0
    max_dd     = float(original.get("max_dd", 0))
    win_rate   = float(original.get("win_rate", 0))
    pf         = original.get("profit_factor")
    n_trades   = int(original.get("total_trades", 0))
    avg_win    = float(original.get("avg_win", 0))
    avg_loss   = float(original.get("avg_loss", 0))
    expectancy = original.get("expectancy") or (
        (win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss) if win_rate else 0
    )

    # Usa reshuffle como referência principal do MC
    ref = mc_results.get("reshuffle") or next(iter(mc_results.values()), {})
    fe  = ref.get("final_equity_pct", {})
    dd  = ref.get("max_drawdown", {})
    sh  = ref.get("sharpe", {})

    p_value   = perm_result.get("p_value", 1.0)
    perm_rank = perm_result.get("backtest_rank", 0.0)
    perm_med  = perm_result.get("median_random", 0.0)
    perm_conc = perm_result.get("conclusion", "REPROVADO")

    ruin_prob      = ref.get("ruin_prob", 0.0)
    rank_equity    = ref.get("backtest_rank_equity", 50.0)
    rank_sharpe    = ref.get("backtest_rank_sharpe", 50.0)
    p95_dd         = dd.get("p95", max_dd)
    dd_ratio       = abs(p95_dd / max_dd) if max_dd != 0 else 0.0

    sep  = "═" * 50
    sep2 = "─" * 50

    lines = [
        sep,
        "STRATEGY VALIDATION REPORT",
        sep,
        f"STRATEGY: {strategy_label}",
        f"ASSET: {asset_label}  |  PERIOD: {period_label}",
        f"TRADES: {n_trades}  |  WIN RATE: {win_rate:.1f}%",
        sep2,
        "── BACKTEST METRICS " + "─" * 31,
        f"Total Return:        {_fmt_money(final_eq - ic)} ({_fmt_pct(total_ret)})",
        f"Sharpe Ratio:        {float(sharpe):.4f}",
        f"Max Drawdown:        {_fmt_pct(max_dd)}",
        f"Profit Factor:       {float(pf):.2f}" if pf is not None and math.isfinite(float(pf)) else "Profit Factor:       ∞",
        f"Expectancy:          {float(expectancy):.4f}%/trade",
        f"Avg Win / Avg Loss:  {avg_win:+.2f}% / {avg_loss:+.2f}%",
        sep2,
        f"── MONTE CARLO ANALYSIS ({n_sims:,} sims) " + "─" * 15,
        "",
        f"{'':26}{'P5':>8}{'P25':>8}{'P50':>8}{'P75':>8}{'P95':>8}",
        f"{'Final Return (%)':26}"
        + f"{fe.get('p5', 0):>8.2f}"
        + f"{fe.get('p25', 0):>8.2f}"
        + f"{fe.get('p50', 0):>8.2f}"
        + f"{fe.get('p75', 0):>8.2f}"
        + f"{fe.get('p95', 0):>8.2f}",
        f"{'Max Drawdown (%)':26}"
        + f"{dd.get('p5', 0):>8.2f}"
        + f"{dd.get('p25', 0):>8.2f}"
        + f"{dd.get('p50', 0):>8.2f}"
        + f"{dd.get('p75', 0):>8.2f}"
        + f"{dd.get('p95', 0):>8.2f}",
        f"{'Sharpe Ratio':26}"
        + f"{sh.get('p5', 0):>8.2f}"
        + f"{sh.get('p25', 0):>8.2f}"
        + f"{sh.get('p50', 0):>8.2f}"
        + f"{sh.get('p75', 0):>8.2f}"
        + f"{sh.get('p95', 0):>8.2f}",
        "",
        f"Backtest Rank:     {_ordinal(rank_equity)} percentile"
        + ("  (above median ✓)" if rank_equity >= 50 else "  (below median ✗)"),
        f"P95 Drawdown:      {p95_dd:.2f}%  ({dd_ratio:.2f}x backtest DD)",
        f"Ruin Probability:  {ruin_prob:.1f}%",
        sep2,
        f"── PERMUTATION TEST ({perm_result.get('n_perms', 0):,} permutations) " + "─" * 10,
        f"Strategy Sharpe:   {float(sharpe):.4f}",
        f"Median Random:     {perm_med:.4f}",
        f"p-value:           {p_value:.4f}",
        f"Result:            {'✓ STATISTICALLY SIGNIFICANT (p < 0.05)' if perm_result.get('approved') else '✗ NOT SIGNIFICANT (p ≥ 0.05)'}",
        f"Percentile Rank:   {_ordinal(perm_rank)}",
        sep2,
        "── VERDICT " + "─" * 39,
        f"{_check(rank_sharpe >= 50)} Sharpe rank {'acima' if rank_sharpe >= 50 else 'abaixo'} do P50 no Monte Carlo",
        f"{_check(dd_ratio < 2.0)} P95 drawdown {'<' if dd_ratio < 2.0 else '>='} 2× backtest drawdown",
        f"{_check(ruin_prob < 5.0)} Probabilidade de ruína {'<' if ruin_prob < 5.0 else '>='} 5%",
        f"{_check(perm_result.get('approved', False))} Permutation test p-value {'< 0.05' if perm_result.get('approved') else '>= 0.05'}",
        "",
    ]

    # Veredicto final
    n_checks = sum([
        rank_sharpe >= 50,
        dd_ratio < 2.0,
        ruin_prob < 5.0,
        perm_result.get("approved", False),
    ])

    if n_checks == 4:
        verdict = "A estratégia demonstra evidência de edge genuíno. ✓"
    elif n_checks >= 2:
        verdict = f"A estratégia é moderadamente robusta ({n_checks}/4 critérios)."
    else:
        verdict = "A estratégia não demonstra robustez estatística suficiente. ✗"

    lines.append(f"CONCLUSÃO: {verdict}")
    lines.append(sep)

    report_text = "\n".join(lines)
    return report_text


def save(text: str, path: str | None = None) -> None:
    """Salva o relatório em arquivo e imprime no terminal."""
    path = path or str(Path(__file__).parents[1] / "output" / "report.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[Relatório salvo em: {path}]")
