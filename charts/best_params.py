import pandas as pd
import streamlit as st
import plotly.express as px


CATEGORICAL_PARAMS = {
    "ma": "Tipo de Media",
    "saida": "Tipo de Saida",
    "stop": "Tipo de Stop",
    "flat": "Flat",
    "pullback": "Pullback",
    "entry_zone": "Entry Zone",
}

NUMERIC_PARAMS = {
    "periodo": "Periodo da Media",
    "lookback": "Lookback",
    "angulo": "Angulo",
    "banda_pct": "Banda (%)",
    "alvo_fixo_pct": "Alvo Fixo (%)",
    "stop_param": "Stop Param",
}


def render_best_params(df: pd.DataFrame, top_n: int = 20):
    """Frequencia e distribuicao dos parametros nas top-N, com faixas sugeridas."""
    if "score" not in df.columns:
        st.warning("Coluna 'Score' necessaria para analise de parametros.")
        return

    top = df.nlargest(top_n, "score")

    st.subheader(f"Melhores Parametros (Top {top_n} parametros)")
    st.caption("Baseado nas parametros com melhor Score para orientar futuros backtests.")

    # Categoricos
    cat_cols = [c for c in CATEGORICAL_PARAMS if c in top.columns]
    if cat_cols:
        st.markdown("**Parametros Categoricos - Frequencia nas Top Estrategias**")
        cols = st.columns(min(len(cat_cols), 3))
        for i, param in enumerate(cat_cols):
            col = cols[i % 3]
            with col:
                counts = top[param].value_counts()
                if counts.empty:
                    continue
                fig = px.bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    labels={"x": CATEGORICAL_PARAMS[param], "y": "Quantidade"},
                    title=CATEGORICAL_PARAMS[param],
                    template="plotly_white",
                    color=counts.values,
                    color_continuous_scale="Greens",
                )
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    # Numericos
    num_cols = [c for c in NUMERIC_PARAMS if c in top.columns and top[c].notna().any()]
    if num_cols:
        st.markdown("**Parametros Numericos - Distribuicao nas Top Estrategias**")
        cols = st.columns(min(len(num_cols), 3))
        for i, param in enumerate(num_cols):
            col = cols[i % 3]
            with col:
                values = top[param].dropna()
                if values.empty:
                    continue
                fig = px.histogram(
                    x=values,
                    nbins=min(15, len(values.unique())),
                    labels={"x": NUMERIC_PARAMS[param], "y": "Quantidade"},
                    title=NUMERIC_PARAMS[param],
                    template="plotly_white",
                    color_discrete_sequence=["#2e7d32"],
                )
                fig.update_layout(
                    height=300,
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Resumo - Faixas Recomendadas para Proximos Testes**")
    summary_rows = []

    for param in cat_cols:
        counts = top[param].value_counts()
        if counts.empty:
            continue
        best = counts.index[0]
        pct = counts.iloc[0] / len(top) * 100
        summary_rows.append({
            "Parametro": CATEGORICAL_PARAMS[param],
            "Melhor Valor": str(best),
            "Frequencia no Top": f"{pct:.0f}%",
            "Faixa Sugerida": ", ".join(counts.head(3).index.astype(str)),
        })

    for param in num_cols:
        values = top[param].dropna()
        if values.empty:
            continue
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        summary_rows.append({
            "Parametro": NUMERIC_PARAMS[param],
            "Melhor Valor": f"{values.median():.2f} (mediana)",
            "Frequencia no Top": f"{len(values)}/{len(top)}",
            "Faixa Sugerida": f"{q25:.2f} a {q75:.2f}",
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
