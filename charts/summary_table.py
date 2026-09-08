import pandas as pd
import streamlit as st
from config import TOP_N


SORT_OPTIONS = {
    "Score": ("score", False),
    "Retorno (%)": ("return_pct", False),
    "Profit Factor": ("profit_factor", False),
    "Sharpe": ("sharpe", False),
    "Win Rate (%)": ("win_rate_pct", False),
    "Trades": ("trades", False),
    "Menor Drawdown": ("max_dd_pct", False),
    "Maior Drawdown": ("max_dd_pct", True),
}


def render_summary_table(df: pd.DataFrame, top_n: int = TOP_N):
    """Tabela das top-N estrategias. Retorna a linha selecionada, ou None."""
    col_sort, col_order = st.columns([2, 1])
    with col_sort:
        sort_label = st.selectbox(
            "Ordenar por:",
            options=list(SORT_OPTIONS.keys()),
            index=0,
        )
    with col_order:
        ascending_toggle = st.toggle("Ordem crescente", value=False)

    sort_col, default_asc = SORT_OPTIONS[sort_label]

    if sort_col not in df.columns:
        st.warning(f"Coluna '{sort_col}' nao encontrada.")
        return None

    ascending = ascending_toggle or default_asc
    top = df.sort_values(sort_col, ascending=ascending).head(top_n)

    display_cols = [
        "ativo", "rank", "return_pct", "max_dd_pct", "trades", "win_rate_pct",
        "profit_factor", "sharpe", "score", "ma", "periodo", "lookback",
    ]
    available = [c for c in display_cols if c in top.columns]
    top = top[available].reset_index(drop=True)

    rename = {
        "ativo": "Ativo",
        "rank": "Rank",
        "return_pct": "Retorno (%)",
        "max_dd_pct": "Max DD (%)",
        "trades": "Trades",
        "win_rate_pct": "Win Rate (%)",
        "profit_factor": "Profit Factor",
        "sharpe": "Sharpe",
        "score": "Score",
        "ma": "MA",
        "periodo": "Periodo",
        "lookback": "Lookback",
    }
    top_display = top.rename(columns={k: v for k, v in rename.items() if k in top.columns})

    def highlight_returns(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #2e7d32"
            elif val < 0:
                return "color: #c62828"
        return ""

    styled = top_display.style.map(
        highlight_returns, subset=["Retorno (%)", "Max DD (%)"]
    )
    styled = styled.format(precision=2, na_rep="-")

    # Gradiente na coluna de ordenacao
    display_sort_col = rename.get(sort_col, sort_col)
    if display_sort_col in top_display.columns:
        styled = styled.background_gradient(
            subset=[display_sort_col],
            cmap="RdYlGn",
            vmin=top_display[display_sort_col].min(),
            vmax=top_display[display_sort_col].max(),
        )

    event = st.dataframe(
        styled,
        use_container_width=True,
        height=min(40 * len(top_display) + 40, 800),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event and event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        rank_val = top.iloc[selected_idx]["rank"]
        original_row = df[df["rank"] == rank_val].iloc[0]
        return original_row

    return None
