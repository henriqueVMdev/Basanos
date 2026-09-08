import pandas as pd
import streamlit as st
from config import COLUMN_DISPLAY


def _copy_params_to_backtest(row: pd.Series):
    """Publica os parametros da linha no session_state do Backtesting Live."""
    if "ma" in row.index and pd.notna(row["ma"]):
        st.session_state["bt_ma_type"] = str(row["ma"]).upper()

    if "periodo" in row.index and pd.notna(row["periodo"]):
        st.session_state["bt_ma_length"] = int(row["periodo"])

    if "lookback" in row.index and pd.notna(row["lookback"]):
        st.session_state["bt_lookback"] = int(row["lookback"])

    if "angulo" in row.index and pd.notna(row["angulo"]):
        val = float(row["angulo"])
        st.session_state["bt_th_up"] = abs(val)
        st.session_state["bt_th_dn"] = -abs(val)

    if "saida" in row.index and pd.notna(row["saida"]):
        saida = str(row["saida"])
        mode_map = {
            "Banda": "Banda + Tendência",
            "Banda + Tendência": "Banda + Tendência",
            "Tendência": "Somente Tendência",
            "Somente Tendência": "Somente Tendência",
            "Alvo Fixo": "Alvo Fixo + Tendência",
            "Alvo Fixo + Tendência": "Alvo Fixo + Tendência",
        }
        st.session_state["bt_exit_mode"] = mode_map.get(saida, saida)

    if "banda_pct" in row.index and pd.notna(row["banda_pct"]):
        val = float(row["banda_pct"])
        st.session_state["bt_pct_up"] = val
        st.session_state["bt_pct_dn"] = val

    if "alvo_fixo_pct" in row.index and pd.notna(row["alvo_fixo_pct"]):
        st.session_state["bt_alvo_fixo"] = float(row["alvo_fixo_pct"])

    if "flat" in row.index and pd.notna(row["flat"]):
        val = row["flat"]
        st.session_state["bt_exit_on_flat"] = val in [True, 1, "True", "true", "Sim", "sim", "S", "s", "1"]

    if "stop" in row.index and pd.notna(row["stop"]):
        stop_val = str(row["stop"])
        if stop_val.lower() in ["off", "false", "0", "nao", "não", ""]:
            st.session_state["bt_use_stop"] = False
        else:
            st.session_state["bt_use_stop"] = True
            st.session_state["bt_stop_type"] = stop_val

    if "stop_param" in row.index and pd.notna(row["stop_param"]):
        st.session_state["bt_stop_param"] = float(row["stop_param"])

    if "pullback" in row.index and pd.notna(row["pullback"]):
        val = row["pullback"]
        st.session_state["bt_use_pullback"] = val in [True, 1, "True", "true", "Sim", "sim", "S", "s", "1"]

    if "entry_zone" in row.index and pd.notna(row["entry_zone"]):
        val = row["entry_zone"]
        st.session_state["bt_use_entry_zone"] = val in [True, 1, "True", "true", "Sim", "sim", "S", "s", "1"]

    st.session_state["bt_params_copied"] = True


def render_strategy_detail(row: pd.Series):
    """Exibe os parametros da estrategia em tres colunas."""
    rank_label = ""
    if "rank" in row.index:
        rank_label = f" - Rank {int(row['rank'])}"

    header_col, btn_col = st.columns([4, 1])
    with header_col:
        st.subheader(f"Detalhes do Parametro{rank_label}")
    with btn_col:
        if st.button("Enviar para Backtesting"):
            _copy_params_to_backtest(row)
            st.success("Parametros copiados para Backtesting Live")

    col1, col2, col3 = st.columns(3)

    items = []
    for col_internal, value in row.items():
        display_name = COLUMN_DISPLAY.get(col_internal, col_internal)
        if pd.isna(value):
            value = "-"
        elif isinstance(value, float):
            value = f"{value:.4f}"
        items.append((display_name, value))

    per_col = (len(items) + 2) // 3
    for i, col in enumerate([col1, col2, col3]):
        start = i * per_col
        chunk = items[start:start + per_col]
        with col:
            for name, val in chunk:
                st.markdown(f"**{name}:** {val}")
