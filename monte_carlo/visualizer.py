"""
Visualizer — geração de gráficos com Matplotlib (dark theme profissional).
Salva PNGs em output/.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paleta
BLUE   = "#378ADD"
GREEN  = "#1D9E75"
RED    = "#E24B4A"
GRAY   = "#888888"
WHITE  = "#E0E0E0"
BG     = "#0d0d12"
PANEL  = "#12121a"
GRID   = (1, 1, 1, 0.06)

OUTPUT = "monte_carlo_project/output"


def _dark_fig(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=10)
    ax.spines[:].set_color("#222230")
    ax.grid(color=GRID, linewidth=0.6, linestyle="--")
    ax.yaxis.label.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    return fig, ax


def _dark_fig2x2(figsize=(16, 12)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=9)
        ax.spines[:].set_color("#222230")
        ax.grid(color=GRID, linewidth=0.5, linestyle="--")
        ax.yaxis.label.set_color(WHITE)
        ax.xaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
    return fig, axes


def _save(fig, name: str):
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvo: {path}")
    return path


# a) Broom Chart
def broom_chart(mc_result: dict, n_sims: int) -> str:
    """Gráfico de vassoura: 100 equity curves simuladas + percentis + original."""
    fig, ax = _dark_fig()

    x = list(range(len(mc_result["p50"])))

    # 100 caminhos (cinza transparente)
    for path in mc_result["sample_paths"]:
        ax.plot(x, path[:len(x)], color=GRAY, alpha=0.07, linewidth=0.6)

    # Bandas
    ax.plot(x, mc_result["p95"][:len(x)], color=GREEN, linewidth=1.2,
            linestyle="--", label="P95", alpha=0.8)
    ax.plot(x, mc_result["p50"][:len(x)], color=WHITE, linewidth=1.5,
            linestyle="--", label="P50 (mediana)", alpha=0.7)
    ax.plot(x, mc_result["p5"][:len(x)],  color=RED,   linewidth=1.2,
            linestyle="--", label="P5",  alpha=0.8)

    # Original
    orig = mc_result["original_equity"]
    ax.plot(range(len(orig)), orig, color=BLUE, linewidth=2.0, label="Backtest original", zorder=5)

    ax.set_xlabel("Nº do Trade" if mc_result.get("x_type") == "trade_index" else "Período", fontsize=11)
    ax.set_ylabel("Capital ($)", fontsize=11)
    ax.set_title(f"Monte Carlo Simulation — Equity Curves (N={n_sims:,})", fontsize=14, fontweight="bold")
    ax.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=10)

    return _save(fig, "broom_chart.png")


# b) Drawdown Distribution
def drawdown_distribution(mc_result: dict, orig_dd: float) -> str:
    fig, ax = _dark_fig()

    dds = mc_result["max_drawdown_dist"]
    p95 = mc_result["max_drawdown"]["p95"]

    ax.hist(dds, bins=60, color=RED, alpha=0.6, edgecolor=BG, linewidth=0.3)
    ax.axvline(orig_dd, color=BLUE, linewidth=2,  linestyle="--", label=f"Backtest: {orig_dd:.2f}%")
    ax.axvline(p95,     color=RED,  linewidth=1.5, linestyle=":",  label=f"P95: {p95:.2f}%")

    ax.annotate(f"Backtest\n{orig_dd:.2f}%", xy=(orig_dd, ax.get_ylim()[1] * 0.9),
                color=BLUE, fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=BLUE, alpha=0.8))

    ax.set_xlabel("Max Drawdown (%)", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.set_title("Distribution of Maximum Drawdowns", fontsize=14, fontweight="bold")
    ax.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=10)

    return _save(fig, "drawdown_distribution.png")


# c) Sharpe Distribution
def sharpe_distribution(mc_result: dict, orig_sharpe: float) -> str:
    fig, ax = _dark_fig()

    sharpes = mc_result["sharpe_dist"]
    rank    = mc_result["backtest_rank_sharpe"]

    ax.hist(sharpes, bins=60, color=BLUE, alpha=0.6, edgecolor=BG, linewidth=0.3)
    ax.axvline(orig_sharpe, color=GREEN, linewidth=2, linestyle="--",
               label=f"Backtest Sharpe: {orig_sharpe:.4f}  ({rank:.1f}º percentil)")

    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.set_title("Distribution of Sharpe Ratios", fontsize=14, fontweight="bold")
    ax.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=10)

    return _save(fig, "sharpe_distribution.png")


# d) Permutation Test
def permutation_test_chart(perm_result: dict) -> str:
    fig, ax = _dark_fig()

    dist     = perm_result["sharpe_dist"]
    orig     = perm_result["original_sharpe"]
    p_value  = perm_result["p_value"]
    concl    = perm_result["conclusion"]
    n_perms  = perm_result["n_perms"]

    ax.hist(dist, bins=50, color=GRAY, alpha=0.6, edgecolor=BG, linewidth=0.3, label="Séries aleatórias")
    ax.axvline(orig, color=BLUE, linewidth=2, linestyle="--", label=f"Estratégia original: {orig:.4f}")

    # Área sombreada: distribuição ≥ original (representa p-value)
    arr_np = np.array(dist)
    hist_vals, bin_edges = np.histogram(arr_np, bins=50)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if lo >= orig:
            ax.axvspan(lo, hi, alpha=0.3, color=RED)

    color_c = GREEN if perm_result.get("approved") else RED
    ax.text(0.97, 0.93, f"p-value = {p_value:.4f}\n{concl}", transform=ax.transAxes,
            ha="right", va="top", color=color_c, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=color_c, alpha=0.9))

    red_patch = mpatches.Patch(color=RED, alpha=0.5, label=f"p-value = {p_value:.4f}")

    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.set_title(f"Permutation Test — Strategy vs Random (N={n_perms:,})", fontsize=14, fontweight="bold")
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], red_patch],
              facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=10)

    return _save(fig, "permutation_test.png")


# e) Summary Dashboard (2×2)
def summary_dashboard(
    mc_result: dict,
    perm_result: dict,
    orig_sharpe: float,
    orig_dd: float,
    n_sims: int,
) -> str:
    fig, axes = _dark_fig2x2()
    fig.suptitle("Strategy Validation Report", fontsize=16, fontweight="bold", color=WHITE, y=1.01)

    ax1, ax2, ax3, ax4 = axes.flat

    # Subplot 1 — Equity curve + bandas
    x = list(range(len(mc_result["p50"])))
    ax1.fill_between(x, mc_result["p5"][:len(x)], mc_result["p95"][:len(x)],
                     color=BLUE, alpha=0.12, label="Banda P5–P95")
    ax1.plot(x, mc_result["p5"][:len(x)],  color=RED,   linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.plot(x, mc_result["p95"][:len(x)], color=GREEN, linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.plot(x, mc_result["p50"][:len(x)], color=WHITE, linewidth=1.0, linestyle="--", alpha=0.6, label="P50")
    orig = mc_result["original_equity"]
    ax1.plot(range(len(orig)), orig, color=BLUE, linewidth=1.8, label="Original", zorder=5)
    ax1.set_title("Equity Curves — Percentile Bands", fontsize=11, fontweight="bold")
    ax1.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=8)

    # Subplot 2 — Drawdown histogram
    p95_dd = mc_result["max_drawdown"]["p95"]
    ax2.hist(mc_result["max_drawdown_dist"], bins=40, color=RED, alpha=0.6, edgecolor=BG, linewidth=0.2)
    ax2.axvline(orig_dd, color=BLUE,  linewidth=1.5, linestyle="--", label=f"Original: {orig_dd:.2f}%")
    ax2.axvline(p95_dd,  color=RED,   linewidth=1.2, linestyle=":",  label=f"P95: {p95_dd:.2f}%")
    ax2.set_title("Distribution of Max Drawdowns", fontsize=11, fontweight="bold")
    ax2.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=8)

    # Subplot 3 — Sharpe MC
    ax3.hist(mc_result["sharpe_dist"], bins=40, color=BLUE, alpha=0.6, edgecolor=BG, linewidth=0.2)
    ax3.axvline(orig_sharpe, color=GREEN, linewidth=1.5, linestyle="--",
                label=f"Original: {orig_sharpe:.4f}")
    ax3.set_title("Sharpe Distribution (Monte Carlo)", fontsize=11, fontweight="bold")
    ax3.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=8)

    # Subplot 4 — Sharpe Permutation Test
    p_value = perm_result["p_value"]
    ax4.hist(perm_result["sharpe_dist"], bins=40, color=GRAY, alpha=0.6, edgecolor=BG, linewidth=0.2)
    ax4.axvline(perm_result["original_sharpe"], color=BLUE, linewidth=1.5,
                linestyle="--", label=f"Original: {perm_result['original_sharpe']:.4f}")
    color_c = GREEN if perm_result.get("approved") else RED
    ax4.text(0.96, 0.92, f"p = {p_value:.4f}\n{perm_result['conclusion']}",
             transform=ax4.transAxes, ha="right", color=color_c,
             fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", fc=BG, ec=color_c, alpha=0.8))
    ax4.set_title("Sharpe Distribution (Permutation Test)", fontsize=11, fontweight="bold")
    ax4.legend(facecolor=BG, edgecolor="#333", labelcolor=WHITE, fontsize=8)

    plt.tight_layout()
    return _save(fig, "summary_dashboard.png")
