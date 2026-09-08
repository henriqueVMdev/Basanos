"""
Otimizador de Parâmetros — Estratégia DePaula v2

Testa múltiplas combinações de parâmetros e rankeia pelos melhores resultados.

Uso:
    python optimizer.py                              # otimiza com dados BTCUSDT
    python optimizer.py --csv dados.csv              # usa CSV local
    python optimizer.py --symbol BTCUSDT --top 20    # mostra top 20 resultados
    python optimizer.py --mode rapido                # grid menor (mais rápido)
    python optimizer.py --mode completo              # grid maior (mais lento)
"""

import itertools
import numpy as np
import pandas as pd
from dataclasses import asdict
from typing import Dict, List, Any
import argparse
import time

from engine.backtesting import (
    Config, run_backtest, download_data, load_csv,
    prepare_indicators, BacktestState,
)


# 1. GRADE DE PARÂMETROS
GRID_RAPIDO = {
    "ma_type":      ["HMA", "EMA"],
    "ma_length":    [21, 50, 100],
    "lookback":     [3, 5, 8],
    "th_up":        [0.3, 0.5, 1.0],
    "exit_mode":    ["Banda + Tendência", "Somente Tendência"],
    "pct_up":       [2.0, 3.0, 5.0],
    "exit_on_flat": [True, False],
    "use_stop":     [False, True],
    "stop_type":    ["ATR"],
    "stop_atr_mult": [1.5, 2.0, 3.0],
    "use_pullback": [True, False],
    "use_norm":     [True, False],
    "atr_length":   [14],
    "hysteresis":   [0.1, 0.2, 0.5],
}

GRID_COMPLETO = {
    "ma_type":      ["SMA", "EMA", "HMA", "WMA"],
    "ma_length":    [14, 21, 34, 50, 100, 200],
    "lookback":     [3, 5, 8, 13],
    "th_up":        [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    "exit_mode":    ["Banda + Tendência", "Somente Tendência", "Alvo Fixo + Tendência"],
    "pct_up":       [1.5, 2.0, 3.0, 5.0, 8.0],
    "alvo_fixo":    [3.0, 5.0, 8.0, 10.0],
    "exit_on_flat": [True, False],
    "use_stop":     [False, True],
    "stop_type":    ["ATR", "Fixo (%)", "Banda Stop"],
    "stop_atr_mult": [1.0, 1.5, 2.0, 3.0],
    "stop_fixo_pct": [1.0, 2.0, 3.0],
    "stop_band_pct": [1.0, 1.5, 2.0],
    "use_pullback": [True, False],
    "use_entry_zone": [False, True],
    "use_norm":     [True, False],
    "atr_length":   [10, 14, 21],
    "hysteresis":   [0.0, 0.1, 0.2, 0.5],
}

GRID_CUSTOM = {
    "ma_type":      ["EMA", "SMA", "HMA"],
    "ma_length":    list(range(40, 150, 5)),
    "lookback":     [0, 5],
    "th_up":        [0.1, 0.5, 1.5, 2.0, 2.5],
    "exit_mode":    ["Banda + Tendência", "Somente Tendência"],
    "pct_up":       list(range(5, 40, 2)),
    "exit_on_flat": [False, True],
    "use_stop":     [True],
    "stop_type":    ["Banda Stop"],
    "stop_band_pct": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "use_pullback": [True, False],
    "use_entry_zone": [True],
    "use_norm":     [False],
    "atr_length":   [14],
    "hysteresis":   [0.2],
}


# 2. GERAÇÃO DE COMBINAÇÕES
def generate_configs(grid: Dict[str, List], base_cfg: Config) -> List[Config]:
    """Gera todas as combinações válidas de parâmetros."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # Simetria: th_dn = -th_up
        if "th_up" in params:
            params["th_dn"] = -params["th_up"]

        # Simetria: pct_dn = pct_up
        if "pct_up" in params:
            params["pct_dn"] = params["pct_up"]

        # Filtra combinações inválidas
        if not _is_valid(params):
            continue

        cfg = Config(**{**asdict(base_cfg), **params})
        configs.append(cfg)

    return configs


def _is_valid(params: dict) -> bool:
    """Remove combinações que não fazem sentido."""

    # Se stop está desligado, não precisa testar variações de stop
    if not params.get("use_stop", False):
        if params.get("stop_type", "ATR") != "ATR":
            return False
        if params.get("stop_atr_mult", 2.0) != 2.0:
            return False
        if params.get("stop_fixo_pct", 2.0) != 2.0:
            return False
        if params.get("stop_band_pct", 1.5) != 1.5:
            return False

    # Se stop é ATR, não varia fixo/banda
    stop_type = params.get("stop_type", "ATR")
    if params.get("use_stop", False):
        if stop_type != "ATR" and params.get("stop_atr_mult", 2.0) != 2.0:
            return False
        if stop_type != "Fixo (%)" and params.get("stop_fixo_pct", 2.0) != 2.0:
            return False
        if stop_type != "Banda Stop" and params.get("stop_band_pct", 1.5) != 1.5:
            return False

    # Se saída é "Somente Tendência", não varia pct/alvo
    exit_mode = params.get("exit_mode", "Banda + Tendência")
    if exit_mode == "Somente Tendência":
        if params.get("pct_up", 3.0) != 3.0:
            return False
        if params.get("alvo_fixo", 5.0) != 5.0:
            return False

    # Se saída é "Banda", não varia alvo fixo
    if exit_mode == "Banda + Tendência":
        if params.get("alvo_fixo", 5.0) != 5.0:
            return False

    # Se saída é "Alvo Fixo", não varia banda
    if exit_mode == "Alvo Fixo + Tendência":
        if params.get("pct_up", 3.0) != 3.0:
            return False

    return True


# 3. MÉTRICAS
def calc_metrics(st: BacktestState, cfg: Config) -> Dict[str, Any]:
    """Calcula métricas de performance para ranking."""
    trades = st.trades
    if not trades:
        return None

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (st.equity / cfg.initial_capital - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

    # Max Drawdown
    eq = st._df["Equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()

    # Periodo total em dias (usado para anualizacao)
    total_days = max((st._df.index[-1] - st._df.index[0]).days, 1) if len(st._df) > 1 else 1

    # Sharpe from equity curve returns (annualized, ddof=1)
    bars_per_year = len(eq) / (total_days / 365.25) if total_days > 0 else 252
    sharpe = 0.0
    if len(eq) > 2:
        eq_rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
        eq_rets = eq_rets[np.isfinite(eq_rets)]
        eq_std = float(eq_rets.std(ddof=1))
        if len(eq_rets) > 1 and eq_std > 0:
            sharpe = float(eq_rets.mean() / eq_std * np.sqrt(bars_per_year))

    # Sortino — annualized by trades_per_year
    neg = [p for p in pnls if p < 0]
    ds_sq = [min(p, 0) ** 2 for p in pnls]
    ds_dev = float(np.sqrt(np.mean(ds_sq))) if ds_sq else 0
    trades_per_year = len(pnls) / (total_days / 365.25) if total_days > 0 else len(pnls)
    sortino = float(np.mean(pnls) / ds_dev * np.sqrt(trades_per_year)) if ds_dev > 0 else 0

    # Calmar — CAGR / |max_dd|
    cagr = ((st.equity / cfg.initial_capital) ** (365.25 / total_days) - 1) * 100
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0

    # Omega (soma ganhos / soma perdas)
    gains_sum = sum(p for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p < 0))
    omega = float(gains_sum / losses_sum) if losses_sum > 0 else 0

    # Sterling (CAGR / media dos N piores drawdown troughs)
    neg_sorted = sorted(neg)
    n_w = min(5, len(neg_sorted))
    if n_w > 0:
        avg_worst = abs(float(np.mean(neg_sorted[:n_w])))
        sterling = float(cagr / avg_worst) if avg_worst > 0 else 0
        burke_d = float(np.sqrt(np.sum(np.array(neg_sorted[:n_w]) ** 2)))
        burke = float(cagr / burke_d) if burke_d > 0 else 0
    else:
        sterling = 0
        burke = 0

    # Score composto: penaliza drawdown, valoriza consistência
    score = total_return * (win_rate / 100) / max(abs(max_dd), 1)

    return {
        "Retorno (%)": round(total_return, 2),
        "Max DD (%)": round(max_dd, 2),
        "Trades": len(trades),
        "Win Rate (%)": round(win_rate, 1),
        "Avg Win (%)": round(avg_win, 2),
        "Avg Loss (%)": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Omega": round(omega, 2),
        "Sterling": round(sterling, 2),
        "Burke": round(burke, 2),
        "Score": round(score, 2),
        # Parâmetros
        "MA": cfg.ma_type,
        "Período": cfg.ma_length,
        "Lookback": cfg.lookback,
        "Ângulo": cfg.th_up,
        "Saída": cfg.exit_mode,
        "Banda (%)": cfg.pct_up,
        "Alvo Fixo (%)": cfg.alvo_fixo,
        "Flat": cfg.exit_on_flat,
        "Stop": f"{cfg.stop_type}" if cfg.use_stop else "OFF",
        "Stop Param": (
            cfg.stop_atr_mult if cfg.stop_type == "ATR" else
            cfg.stop_fixo_pct if cfg.stop_type == "Fixo (%)" else
            cfg.stop_band_pct
        ) if cfg.use_stop else 0,
        "Pullback": cfg.use_pullback,
        "Entry Zone": cfg.use_entry_zone,
        "Norm ATR": cfg.use_norm,
        "ATR Length": cfg.atr_length,
        "Histerese": cfg.hysteresis,
    }


# 4. OTIMIZADOR
def optimize(df: pd.DataFrame, grid: Dict, base_cfg: Config, rank_by: str = "Score") -> pd.DataFrame:
    """Roda backtest para todas as combinações e retorna DataFrame rankeado."""

    configs = generate_configs(grid, base_cfg)
    total = len(configs)
    print(f"\n  Combinações a testar: {total}")

    if total > 50000:
        print(f"  AVISO: {total} combinações — pode demorar ~{total * 0.05:.0f}s")

    results = []
    t0 = time.time()
    last_print = t0

    for i, cfg in enumerate(configs):
        try:
            st = run_backtest(df.copy(), cfg)
            metrics = calc_metrics(st, cfg)
            if metrics and metrics["Trades"] >= 5:  # Mínimo 5 trades
                results.append(metrics)
        except Exception:
            pass

        # Progresso
        now = time.time()
        if now - last_print >= 2.0 or i == total - 1:
            elapsed = now - t0
            pct = (i + 1) / total * 100
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            print(f"\r  Progresso: {i+1}/{total} ({pct:.1f}%) | "
                  f"Válidos: {len(results)} | "
                  f"Tempo: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)
            last_print = now

    elapsed = time.time() - t0
    print(f"\n  Concluído em {elapsed:.1f}s | {len(results)} configurações válidas\n")

    if not results:
        print("  Nenhum resultado válido encontrado.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    results_df.index += 1  # Ranking começa em 1

    return results_df


# 5. DISPLAY
def print_top_results(results_df: pd.DataFrame, top_n: int = 10, rank_by: str = "Score"):
    """Imprime os melhores resultados formatados."""

    if results_df.empty:
        return

    display = results_df.head(top_n)

    print("=" * 120)
    print(f"  TOP {top_n} CONFIGURAÇÕES (rankeado por {rank_by})")
    print("=" * 120)

    cols_metrics = ["Retorno (%)", "Max DD (%)", "Trades", "Win Rate (%)", "Profit Factor", "Sharpe", "Score"]
    cols_params = ["MA", "Período", "Lookback", "Ângulo", "Saída", "Banda (%)", "Flat", "Stop", "Pullback"]

    # Métricas
    print(f"\n  {'#':>3}  {'Retorno':>10}  {'MaxDD':>8}  {'Trades':>6}  {'WinRate':>8}  {'PF':>6}  {'Sharpe':>7}  {'Score':>8}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*8}")

    for idx, row in display.iterrows():
        print(f"  {idx:>3}  {row['Retorno (%)']:>+10.2f}  {row['Max DD (%)']:>8.2f}  "
              f"{row['Trades']:>6}  {row['Win Rate (%)']:>7.1f}%  {row['Profit Factor']:>6.2f}  "
              f"{row['Sharpe']:>7.2f}  {row['Score']:>8.2f}")

    # Parâmetros das top configs
    print(f"\n  {'#':>3}  {'MA':<5} {'Per':>4} {'LB':>3} {'Âng':>5}  {'Saída':<22} {'Banda':>6} {'Flat':<5} {'Stop':<12} {'PB':<4} {'EZ':<4}")
    print(f"  {'─'*3}  {'─'*5} {'─'*4} {'─'*3} {'─'*5}  {'─'*22} {'─'*6} {'─'*5} {'─'*12} {'─'*4} {'─'*4}")

    for idx, row in display.iterrows():
        stop_str = row["Stop"]
        if stop_str != "OFF":
            stop_str = f"{stop_str}({row['Stop Param']})"

        print(f"  {idx:>3}  {row['MA']:<5} {row['Período']:>4} {row['Lookback']:>3} {row['Ângulo']:>5.1f}  "
              f"{row['Saída']:<22} {row['Banda (%)']:>5.1f}% {'SIM' if row['Flat'] else 'NÃO':<5} "
              f"{stop_str:<12} {'SIM' if row['Pullback'] else 'NÃO':<4} "
              f"{'SIM' if row['Entry Zone'] else 'NÃO':<4}")

    print("\n" + "=" * 120)

    # Melhor configuração como comando
    best = display.iloc[0]
    stop_args = ""
    if best["Stop"] != "OFF":
        stop_args = f" --use-stop --stop-type \"{best['Stop']}\""
        if best["Stop"] == "ATR":
            stop_args += f" --stop-atr-mult {best['Stop Param']}"
        elif best["Stop"] == "Fixo (%)":
            stop_args += f" --stop-fixo-pct {best['Stop Param']}"
        elif best["Stop"] == "Banda Stop":
            stop_args += f" --stop-band-pct {best['Stop Param']}"

    print(f"\n  Melhor configuração como comando:")
    print(f"  python backtesting.py"
          f" --ma-type {best['MA']}"
          f" --ma-length {best['Período']}"
          f" --lookback {best['Lookback']}"
          f" --th-up {best['Ângulo']}"
          f" --exit-mode \"{best['Saída']}\""
          f" --pct-up {best['Banda (%)']}"
          f" {'--exit-on-flat' if best['Flat'] else '--no-exit-on-flat'}"
          f"{stop_args}"
          f" {'--use-pullback' if best['Pullback'] else '--no-pullback'}"
          f" {'--use-entry-zone' if best['Entry Zone'] else ''}"
    )
    print()


# 6. MAIN
def main():
    parser = argparse.ArgumentParser(description="Otimizador — Estratégia DePaula v2")

    parser.add_argument("--csv", type=str, help="Caminho do CSV")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo (default: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1d", help="Intervalo (default: 1d)")
    parser.add_argument("--mode", type=str, default="rapido", choices=["rapido", "completo", "custom"],
                        help="Tamanho do grid de busca")
    parser.add_argument("--top", type=int, default=10, help="Quantos resultados mostrar (default: 10)")
    parser.add_argument("--rank-by", type=str, default="Score",
                        choices=["Score", "Retorno (%)", "Sharpe", "Sortino", "Calmar", "Omega", "Sterling", "Burke", "Profit Factor", "Win Rate (%)"],
                        help="Métrica para ranking (default: Score)")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--export", type=str, default="otimizacao_resultado.csv",
                        help="Arquivo CSV de saída com todos os resultados")

    args = parser.parse_args()

    # Dados
    if args.csv:
        df = load_csv(args.csv)
    else:
        df = download_data(args.symbol, args.interval)

    # Grid
    grids = {
        "rapido": GRID_RAPIDO,
        "completo": GRID_COMPLETO,
        "custom": GRID_CUSTOM,
    }
    grid = grids[args.mode]

    # Config base
    base_cfg = Config(
        initial_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Otimiza
    results_df = optimize(df, grid, base_cfg, rank_by=args.rank_by)

    if not results_df.empty:
        # Mostra top resultados
        print_top_results(results_df, top_n=args.top, rank_by=args.rank_by)

        # Exporta todos os resultados (CSV padrão com vírgula + BOM para Excel)
        results_df.to_csv(args.export, index=True, index_label="Rank",
                          sep=",", encoding="utf-8-sig")
        print(f"  Todos os resultados exportados para: {args.export}")

        # Exporta também em Excel (.xlsx)
        try:
            excel_path = args.export.replace(".csv", ".xlsx")
            results_df.to_excel(excel_path, index=True, index_label="Rank", engine="openpyxl")
            print(f"  Resultados exportados em Excel: {excel_path}")
        except ImportError:
            print("  [Aviso] openpyxl não instalado — Excel não exportado. Instale com: pip install openpyxl")
        print(f"  Total de configurações testadas: {len(results_df)}")


if __name__ == "__main__":
    main()
