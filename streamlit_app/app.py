import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st
import pandas as pd

from config import DATA_DIR, TOP_N
from loader import load_csv
from charts.scatter import plot_return_vs_drawdown, plot_return_vs_sharpe, plot_return_vs_trades
from charts.summary_table import render_summary_table
from charts.strategy_detail import render_strategy_detail
from charts.best_params import render_best_params

# Page config
st.set_page_config(
    page_title="Backtesting Dashboard",
    layout="wide",
)

st.title("Backtesting Strategy Dashboard")

# Sidebar: carga de dados
st.sidebar.header("Dados")

upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])

csv_files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
csv_names = [f.name for f in csv_files]

selected_file = None
if csv_names:
    selected_name = st.sidebar.selectbox(
        "Ou selecione um arquivo existente:",
        options=[""] + csv_names,
    )
    if selected_name:
        selected_file = DATA_DIR / selected_name

# Carregar dados
df = None
try:
    if upload is not None:
        df = load_csv(upload)
        st.session_state["dashboard_df"] = df
        st.session_state["dashboard_file"] = upload.name
        st.sidebar.success(f"CSV carregado: {upload.name} ({len(df)} linhas)")
    elif selected_file is not None:
        df = load_csv(selected_file)
        st.session_state["dashboard_df"] = df
        st.session_state["dashboard_file"] = selected_file.name
        st.sidebar.success(f"CSV carregado: {selected_file.name} ({len(df)} linhas)")
except Exception as e:
    st.error(f"Erro ao carregar CSV: {e}")

# Recuperar dados da sessao se nenhum arquivo novo foi selecionado
if df is None and "dashboard_df" in st.session_state:
    df = st.session_state["dashboard_df"]
    file_name = st.session_state.get("dashboard_file", "")
    st.sidebar.success(f"CSV em memoria: {file_name} ({len(df)} linhas)")

if df is None:
    st.info(
        "Faca upload de um CSV ou coloque um arquivo na pasta data/ e selecione-o na sidebar."
    )
    st.stop()
    raise SystemExit

# Ativo
if "ativo" in df.columns:
    ativo = df["ativo"].iloc[0]
    st.sidebar.markdown(f"**Ativo:** {ativo}")

# Sidebar: filtros
st.sidebar.header("Filtros")

min_trades = st.sidebar.number_input("Min Trades", min_value=0, value=0, step=1)
sharpe_min_val = float(df["sharpe"].min()) if "sharpe" in df.columns else -10.0
min_sharpe = st.sidebar.number_input("Min Sharpe", value=sharpe_min_val, step=0.1, format="%.2f")

ret_min = float(df["return_pct"].min()) if "return_pct" in df.columns else -100.0
ret_max = float(df["return_pct"].max()) if "return_pct" in df.columns else 100.0
return_range = st.sidebar.slider(
    "Retorno (%)",
    min_value=ret_min,
    max_value=ret_max,
    value=(ret_min, ret_max),
)

top_n = st.sidebar.slider("Top-N na tabela", min_value=5, max_value=100, value=TOP_N, step=5)

# Aplicar filtros
filtered = df.copy()
if "trades" in filtered.columns:
    filtered = filtered[filtered["trades"] >= min_trades]
if "sharpe" in filtered.columns:
    filtered = filtered[filtered["sharpe"] >= min_sharpe]
if "return_pct" in filtered.columns:
    filtered = filtered[
        (filtered["return_pct"] >= return_range[0])
        & (filtered["return_pct"] <= return_range[1])
    ]

st.sidebar.metric("Parametros filtrados", len(filtered))

# Metricas resumo
if "ativo" in df.columns:
    st.subheader(f"Resumo - {df['ativo'].iloc[0]}")
else:
    st.subheader("Resumo")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Parametros", len(filtered))
if "return_pct" in filtered.columns:
    col2.metric("Retorno Medio", f"{filtered['return_pct'].mean():.2f}%")
if "sharpe" in filtered.columns:
    col3.metric("Sharpe Medio", f"{filtered['sharpe'].mean():.2f}")
if "score" in filtered.columns:
    col4.metric("Melhor Score", f"{filtered['score'].max():.2f}")

# Variavel para guardar estrategia selecionada
selected_strategy = None


def _get_strategy_from_chart_event(event, source_df):
    """Extrai a estrategia do evento de selecao do grafico."""
    if event and event.selection and event.selection.points:
        point = event.selection.points[0]
        custom = point.get("customdata")
        if custom:
            rank_val = custom[0]
            match = source_df[source_df["rank"] == rank_val]
            if not match.empty:
                return match.iloc[0]
    return None


# Scatter Plots
st.subheader("Analise Visual")

tab1, tab2, tab3 = st.tabs([
    "Retorno vs Drawdown",
    "Retorno vs Sharpe",
    "Retorno vs Trades",
])

with tab1:
    if all(c in filtered.columns for c in ["max_dd_pct", "return_pct", "score"]):
        event1 = st.plotly_chart(
            plot_return_vs_drawdown(filtered),
            use_container_width=True,
            on_select="rerun",
            key="chart_dd",
        )
        result = _get_strategy_from_chart_event(event1, filtered)
        if result is not None:
            selected_strategy = result
    else:
        st.warning("Colunas necessarias nao encontradas para este grafico.")

with tab2:
    if all(c in filtered.columns for c in ["sharpe", "return_pct", "win_rate_pct"]):
        event2 = st.plotly_chart(
            plot_return_vs_sharpe(filtered),
            use_container_width=True,
            on_select="rerun",
            key="chart_sharpe",
        )
        result = _get_strategy_from_chart_event(event2, filtered)
        if result is not None:
            selected_strategy = result
    else:
        st.warning("Colunas necessarias nao encontradas para este grafico.")

with tab3:
    if all(c in filtered.columns for c in ["trades", "return_pct", "profit_factor"]):
        event3 = st.plotly_chart(
            plot_return_vs_trades(filtered),
            use_container_width=True,
            on_select="rerun",
            key="chart_trades",
        )
        result = _get_strategy_from_chart_event(event3, filtered)
        if result is not None:
            selected_strategy = result
    else:
        st.warning("Colunas necessarias nao encontradas para este grafico.")

# Tabela Top-N
st.subheader(f"Top {top_n} Parametros")
table_result = render_summary_table(filtered, top_n=top_n)
if table_result is not None:
    selected_strategy = table_result

# Detalhes da estrategia selecionada
if selected_strategy is not None:
    st.divider()
    render_strategy_detail(selected_strategy)

# Melhores parametros
st.divider()
render_best_params(filtered, top_n=top_n)
