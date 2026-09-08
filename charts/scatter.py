import numpy as np
import plotly.express as px
import pandas as pd


def _filter_cols(df, cols):
    """Retorna apenas as colunas que existem no dataframe."""
    return [c for c in cols if c in df.columns]


def plot_return_vs_drawdown(df: pd.DataFrame):
    """Scatter: Retorno (%) vs Max Drawdown (%), colorido por Score."""
    plot_df = df.reset_index(drop=True)
    hover = _filter_cols(plot_df, ["rank", "ma", "periodo", "lookback", "profit_factor", "sharpe"])
    fig = px.scatter(
        plot_df,
        x="max_dd_pct",
        y="return_pct",
        color="score",
        color_continuous_scale="RdYlGn",
        custom_data=_filter_cols(plot_df, ["rank"]),
        hover_data=hover,
        labels={
            "max_dd_pct": "Max Drawdown (%)",
            "return_pct": "Retorno (%)",
            "score": "Score",
        },
        title="Retorno vs Max Drawdown",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        height=500,
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Retorno (%)",
    )
    return fig


def plot_return_vs_sharpe(df: pd.DataFrame):
    """Scatter: Retorno (%) vs Sharpe, colorido por Win Rate."""
    plot_df = df.reset_index(drop=True)
    hover = _filter_cols(plot_df, ["rank", "trades", "max_dd_pct", "profit_factor", "ma"])
    fig = px.scatter(
        plot_df,
        x="sharpe",
        y="return_pct",
        color="win_rate_pct",
        color_continuous_scale="Viridis",
        custom_data=_filter_cols(plot_df, ["rank"]),
        hover_data=hover,
        labels={
            "sharpe": "Sharpe Ratio",
            "return_pct": "Retorno (%)",
            "win_rate_pct": "Win Rate (%)",
        },
        title="Retorno vs Sharpe Ratio",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        height=500,
        xaxis_title="Sharpe Ratio",
        yaxis_title="Retorno (%)",
    )
    return fig


def plot_return_vs_trades(df: pd.DataFrame):
    """Scatter: Retorno (%) vs Numero de Trades, colorido por Profit Factor.

    Escala divergente com breakeven (PF=1) fixo no meio e extremos nos
    percentis 5-95, para nao achatar o contraste com outliers.
    """
    plot_df = df.reset_index(drop=True)

    pf_raw = plot_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pf_raw) >= 2:
        pf_lo = float(max(0.0, np.percentile(pf_raw, 5)))
        pf_hi = float(np.percentile(pf_raw, 95))
        if pf_hi - pf_lo < 0.5:
            pf_hi = pf_lo + 0.5
    else:
        pf_lo, pf_hi = 0.0, 3.0

    # Mantem 1.0 dentro do range, garantindo escala estritamente crescente
    pf_lo = min(pf_lo, 0.8)
    pf_hi = max(pf_hi, 1.5)
    PF_SCALE, NORM_SCALE = [pf_lo, 1.0, pf_hi], [0.0, 0.5, 1.0]

    plot_df = plot_df.copy()
    plot_df["pf_norm"] = np.interp(plot_df["profit_factor"], PF_SCALE, NORM_SCALE)

    hover = _filter_cols(plot_df, ["rank", "sharpe", "max_dd_pct", "win_rate_pct", "ma", "profit_factor"])
    hover_data = {c: True for c in hover}
    hover_data["pf_norm"] = False  # campo interno de cor
    fig = px.scatter(
        plot_df,
        x="trades",
        y="return_pct",
        color="pf_norm",
        color_continuous_scale=[
            [0.0,  "#b71c1c"],
            [0.25, "#ef5350"],
            [0.49, "#ffeb3b"],
            [0.51, "#ffeb3b"],
            [0.75, "#66bb6a"],
            [1.0,  "#1b5e20"],
        ],
        range_color=[0.0, 1.0],
        custom_data=_filter_cols(plot_df, ["rank"]),
        hover_data=hover_data,
        labels={
            "trades": "Numero de Trades",
            "return_pct": "Retorno (%)",
            "profit_factor": "Profit Factor",
        },
        title="Retorno vs Numero de Trades",
        template="plotly_white",
    )

    # Colorbar exibe o PF real, desfazendo a normalizacao
    tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig.update_coloraxes(colorbar=dict(
        title="Profit Factor",
        tickvals=tickvals,
        ticktext=[f"{pf:.2f}" for pf in np.interp(tickvals, NORM_SCALE, PF_SCALE)],
    ))
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(
        height=500,
        xaxis_title="Trades",
        yaxis_title="Retorno (%)",
    )
    return fig
                                                                                                                                                                                                            