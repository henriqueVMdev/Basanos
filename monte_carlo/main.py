"""
Entry point — Monte Carlo Strategy Validation (CLI).

Uso:
    python -m monte_carlo.main
    python -m monte_carlo.main --ticker ETH-USD --fast 10 --slow 30 --sims 2000
"""
from __future__ import annotations

import argparse
import sys
import os

# Garante que o root do projeto está no path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from monte_carlo.strategy      import SMACrossover, SMACrossConfig, download_data
from monte_carlo.monte_carlo   import MonteCarlo
from monte_carlo.permutation_test import PermutationTest
from monte_carlo.visualizer    import (
    broom_chart, drawdown_distribution, sharpe_distribution,
    permutation_test_chart, summary_dashboard,
)
from monte_carlo.report import generate as gen_report, save as save_report


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Strategy Validation")
    parser.add_argument("--ticker",  default="BTC-USD", help="Ticker yfinance (default: BTC-USD)")
    parser.add_argument("--period",  default="4y",      help="Período histórico (default: 4y)")
    parser.add_argument("--fast",    type=int, default=20, help="Período MA rápida")
    parser.add_argument("--slow",    type=int, default=50, help="Período MA lenta")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--sims",    type=int, default=5000, help="Nº de simulações MC")
    parser.add_argument("--perms",   type=int, default=1000, help="Nº de permutações")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--no-perm", action="store_true", help="Pula o permutation test (mais rápido)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   Monte Carlo Strategy Validation               ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n[1/5] Baixando dados: {args.ticker} ({args.period})...")

    df = download_data(args.ticker, args.period)
    print(f"      {len(df)} candles carregados ({df.index[0].date()} → {df.index[-1].date()})")

    cfg = SMACrossConfig(
        fast=args.fast,
        slow=args.slow,
        initial_capital=args.capital,
    )
    strat  = SMACrossover()
    result = strat.run(df, cfg)
    trades = result.trades
    metrics = result.metrics
    equity_vals  = result.equity_curve["values"]
    equity_dates = result.equity_curve["dates"]

    if not trades:
        print("ERRO: Nenhum trade gerado. Verifique os parâmetros.")
        sys.exit(1)

    print(f"\n[2/5] Backtest original:")
    print(f"      Trades: {metrics['total_trades']}  |  Return: {metrics['total_return']:.2f}%"
          f"  |  Sharpe: {metrics['sharpe']:.4f}  |  DD: {metrics['max_dd']:.2f}%")

    print(f"\n[3/5] Monte Carlo ({args.sims:,} simulações, seed={args.seed})...")
    mc = MonteCarlo(initial_capital=args.capital, seed=args.seed)

    mc_results = {
        "reshuffle":        mc.reshuffle(trades, n_sims=args.sims),
        "resample":         mc.resample(trades,  n_sims=args.sims),
        "randomized":       mc.randomized(trades, n_sims=args.sims),
        "return_alteration": mc.return_alteration(equity_vals, equity_dates, n_sims=args.sims),
    }
    print("      Concluído.")

    print(f"\n[4/5] Permutation Test ({args.perms} permutações)...")
    if args.no_perm:
        perm_result = {
            "p_value": 1.0, "sharpe_dist": [], "original_sharpe": metrics.get("sharpe", 0),
            "original_pf": metrics.get("profit_factor"), "median_random": 0.0,
            "backtest_rank": 50.0, "conclusion": "N/A", "approved": False, "n_perms": 0,
        }
    else:
        def strategy_fn(df_):
            r = strat.run(df_, cfg)
            return {"metrics": r.metrics, "equity_curve": r.equity_curve, "trades": r.trades}

        pt = PermutationTest(seed=args.seed)
        perm_result = pt.run(df, strategy_fn, n_perms=args.perms)
    print(f"      p-value = {perm_result['p_value']:.4f} → {perm_result['conclusion']}")

    print("\n[5/5] Gerando gráficos e relatório...")
    ref    = mc_results["reshuffle"]
    orig_s = float(metrics.get("sharpe") or 0)
    orig_d = float(metrics.get("max_dd") or 0)

    broom_chart(ref, args.sims)
    drawdown_distribution(ref, orig_d)
    sharpe_distribution(ref, orig_s)
    if not args.no_perm:
        permutation_test_chart(perm_result)
    summary_dashboard(ref, perm_result, orig_s, orig_d, args.sims)

    strat_label  = f"SMA Crossover (fast={args.fast}, slow={args.slow})"
    period_label = f"{df.index[0].date()} → {df.index[-1].date()}"
    report_text  = gen_report(
        original=metrics,
        mc_results=mc_results,
        perm_result=perm_result,
        strategy_label=strat_label,
        asset_label=args.ticker,
        period_label=period_label,
        n_sims=args.sims,
    )
    save_report(report_text)


if __name__ == "__main__":
    main()
