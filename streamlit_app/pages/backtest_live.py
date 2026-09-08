import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Backtesting Live", layout="wide")
st.title("Backtesting Live")

# Importar engine do backtesting
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from engine.backtesting import Config, run_backtest, download_data, load_csv as bt_load_csv

# Lista de ativos pre-definidos
ASSETS = {
    "Indices US": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
    },
    "Acoes US - Tech": {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "NVIDIA (NVDA)": "NVDA",
        "Broadcom (AVGO)": "AVGO",
        "Alphabet (GOOGL)": "GOOGL",
        "Amazon (AMZN)": "AMZN",
        "Meta (META)": "META",
        "Tesla (TSLA)": "TSLA",
        "AMD (AMD)": "AMD",
        "Intel (INTC)": "INTC",
        "Netflix (NFLX)": "NFLX",
        "Salesforce (CRM)": "CRM",
    },
    "Acoes US - Financeiro": {
        "JPMorgan (JPM)": "JPM",
        "Goldman Sachs (GS)": "GS",
        "Bank of America (BAC)": "BAC",
        "Visa (V)": "V",
        "Mastercard (MA)": "MA",
        "Berkshire Hathaway (BRK.B)": "BRK-B",
    },
    "Acoes US - Saude": {
        "UnitedHealth (UNH)": "UNH",
        "Johnson & Johnson (JNJ)": "JNJ",
        "Eli Lilly (LLY)": "LLY",
        "Pfizer (PFE)": "PFE",
        "AbbVie (ABBV)": "ABBV",
    },
    "Acoes US - Energia & Industria": {
        "ExxonMobil (XOM)": "XOM",
        "Chevron (CVX)": "CVX",
        "Boeing (BA)": "BA",
        "Caterpillar (CAT)": "CAT",
        "Lockheed Martin (LMT)": "LMT",
    },
    "Acoes US - Consumo": {
        "Walmart (WMT)": "WMT",
        "Coca-Cola (KO)": "KO",
        "McDonald's (MCD)": "MCD",
        "Nike (NKE)": "NKE",
        "Disney (DIS)": "DIS",
    },
    "Crypto": {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
        "Solana (SOL)": "SOL-USD",
        "BNB": "BNB-USD",
        "XRP": "XRP-USD",
        "Cardano (ADA)": "ADA-USD",
        "Avalanche (AVAX)": "AVAX-USD",
        "Polkadot (DOT)": "DOT-USD",
        "Chainlink (LINK)": "LINK-USD",
        "Polygon (MATIC)": "MATIC-USD",
        "Dogecoin (DOGE)": "DOGE-USD",
        "Litecoin (LTC)": "LTC-USD",
    },
    "Commodities - Metais": {
        "Ouro (Gold)": "GC=F",
        "Prata (Silver)": "SI=F",
        "Cobre (Copper)": "HG=F",
        "Platina (Platinum)": "PL=F",
        "Paladio (Palladium)": "PA=F",
    },
    "Commodities - Energia": {
        "Petroleo WTI": "CL=F",
        "Petroleo Brent": "BZ=F",
        "Gas Natural": "NG=F",
    },
    "Commodities - Agricola": {
        "Milho (Corn)": "ZC=F",
        "Soja (Soybean)": "ZS=F",
        "Trigo (Wheat)": "ZW=F",
        "Cafe (Coffee)": "KC=F",
        "Acucar (Sugar)": "SB=F",
    },
}

# Montar lista flat para o selectbox
asset_options = {}
for category, items in ASSETS.items():
    for label, ticker in items.items():
        asset_options[f"{category} | {label}"] = ticker

# Sidebar: dados
st.sidebar.header("Dados")
data_source = st.sidebar.radio("Fonte de dados:", ["Lista de Ativos", "CSV"])

df_data = None

if data_source == "Lista de Ativos":
    selected_asset = st.sidebar.selectbox("Ativo", options=list(asset_options.keys()))
    symbol = asset_options[selected_asset]
    interval = st.sidebar.selectbox("Timeframe", ["1d", "1h", "4h", "1wk", "1mo"], index=0)
    if st.sidebar.button("Carregar dados"):
        with st.spinner("Baixando dados..."):
            try:
                df_data = download_data(symbol, interval)
                st.session_state["bt_data"] = df_data
                st.session_state["bt_symbol"] = selected_asset.split(" | ")[1]
                st.session_state["bt_interval"] = interval
                st.sidebar.success(f"{selected_asset} | {len(df_data)} barras carregadas")
            except Exception as e:
                st.sidebar.error(f"Erro: {e}")

    if "bt_data" in st.session_state:
        df_data = st.session_state["bt_data"]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (Date,Open,High,Low,Close)", type=["csv"])
    if uploaded is not None:
        try:
            df_data = pd.read_csv(uploaded, parse_dates=["Date"], index_col="Date").sort_index()
            st.session_state["bt_data"] = df_data
            st.session_state["bt_symbol"] = uploaded.name
            st.session_state["bt_interval"] = "-"
            st.sidebar.success(f"{uploaded.name} | {len(df_data)} barras")
        except Exception as e:
            st.sidebar.error(f"Erro: {e}")

    if "bt_data" in st.session_state:
        df_data = st.session_state["bt_data"]

if df_data is None:
    st.info("Selecione um ativo e clique em 'Carregar dados' ou faca upload de um CSV.")
    st.stop()
    raise SystemExit

# Defaults do session_state (copiados do Dashboard)
p = st.session_state

# Flag de parametros copiados
if p.get("bt_params_copied"):
    st.info("Parametros carregados do Dashboard")
    p["bt_params_copied"] = False

# Defaults
MA_OPTIONS = ["SMA", "EMA", "HMA", "RMA", "WMA"]
EXIT_OPTIONS = ["Banda + Tendência", "Somente Tendência", "Alvo Fixo + Tendência"]
STOP_OPTIONS = ["ATR", "Fixo (%)", "Banda Stop"]

def _ma_index():
    val = p.get("bt_ma_type", "HMA")
    return MA_OPTIONS.index(val) if val in MA_OPTIONS else 2

def _exit_index():
    val = p.get("bt_exit_mode", "Banda + Tendência")
    return EXIT_OPTIONS.index(val) if val in EXIT_OPTIONS else 0

def _stop_index():
    val = p.get("bt_stop_type", "ATR")
    return STOP_OPTIONS.index(val) if val in STOP_OPTIONS else 0

# Sidebar: parametros da estrategia
st.sidebar.header("Media Movel")
ma_type = st.sidebar.selectbox("Tipo de Media", MA_OPTIONS, index=_ma_index())
ma_length = st.sidebar.number_input("Periodo", min_value=2, max_value=500, value=int(p.get("bt_ma_length", 50)), step=1)
lookback = st.sidebar.number_input("Lookback", min_value=2, max_value=100, value=int(p.get("bt_lookback", 5)), step=1)
th_up = st.sidebar.number_input("Angulo Alta", value=float(p.get("bt_th_up", 0.5)), step=0.1, format="%.1f")
th_dn = st.sidebar.number_input("Angulo Baixa", value=float(p.get("bt_th_dn", -0.5)), step=0.1, format="%.1f")
hysteresis = st.sidebar.number_input("Histerese", value=0.2, step=0.1, format="%.1f")

st.sidebar.header("Saida")
exit_mode = st.sidebar.selectbox("Modo de Saida", EXIT_OPTIONS, index=_exit_index())
pct_up = st.sidebar.number_input("Banda Superior (%)", value=float(p.get("bt_pct_up", 3.0)), step=0.5, format="%.1f")
pct_dn = st.sidebar.number_input("Banda Inferior (%)", value=float(p.get("bt_pct_dn", 3.0)), step=0.5, format="%.1f")
alvo_fixo = st.sidebar.number_input("Alvo Fixo (%)", value=float(p.get("bt_alvo_fixo", 5.0)), step=0.5, format="%.1f")
exit_on_flat = st.sidebar.toggle("Sair no Flat (cinza)", value=p.get("bt_exit_on_flat", True))

st.sidebar.header("Stop Loss")
use_stop = st.sidebar.toggle("Usar Stop", value=p.get("bt_use_stop", False))
stop_type = st.sidebar.selectbox("Tipo de Stop", STOP_OPTIONS, index=_stop_index())
stop_atr_mult = st.sidebar.number_input("ATR Multiplicador", value=float(p.get("bt_stop_param", 2.0)), step=0.1, format="%.1f")
stop_fixo_pct = st.sidebar.number_input("Stop Fixo (%)", value=2.0, step=0.5, format="%.1f")
stop_band_pct = st.sidebar.number_input("Stop Banda (%)", value=1.5, step=0.5, format="%.1f")

st.sidebar.header("Entrada")
use_pullback = st.sidebar.toggle("Pullback", value=p.get("bt_use_pullback", True))
use_entry_zone = st.sidebar.toggle("Entry Zone", value=p.get("bt_use_entry_zone", False))

st.sidebar.header("Capital")
initial_capital = st.sidebar.number_input("Capital Inicial", value=1000.0, step=100.0, format="%.2f")

# Rodar backtest
cfg = Config(
    ma_type=ma_type,
    ma_length=ma_length,
    lookback=lookback,
    th_up=th_up,
    th_dn=th_dn,
    hysteresis=hysteresis,
    exit_mode=exit_mode,
    pct_up=pct_up,
    pct_dn=pct_dn,
    alvo_fixo=alvo_fixo,
    exit_on_flat=exit_on_flat,
    use_stop=use_stop,
    stop_type=stop_type,
    stop_atr_mult=stop_atr_mult,
    stop_fixo_pct=stop_fixo_pct,
    stop_band_pct=stop_band_pct,
    use_pullback=use_pullback,
    use_entry_zone=use_entry_zone,
    initial_capital=initial_capital,
)

with st.spinner("Executando backtest..."):
    result = run_backtest(df_data.copy(), cfg)

bt_df = result._df
trades = result.trades

# Metricas resumo
symbol_label = st.session_state.get("bt_symbol", "")
interval_label = st.session_state.get("bt_interval", "")
st.subheader(f"Resultado - {symbol_label} ({interval_label})")

total_return = (result.equity / cfg.initial_capital - 1) * 100
wins = [t for t in trades if t.pnl_pct > 0]
losses = [t for t in trades if t.pnl_pct <= 0]
win_rate = len(wins) / len(trades) * 100 if trades else 0
avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
pf_num = sum(t.pnl_pct for t in wins)
pf_den = abs(sum(t.pnl_pct for t in losses))
profit_factor = pf_num / pf_den if pf_den > 0 else float("inf")

eq = bt_df["Equity"].values
peak = np.maximum.accumulate(eq)
dd = (eq - peak) / peak * 100
max_dd = dd.min()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Capital Final", f"${result.equity:,.2f}")
c2.metric("Retorno", f"{total_return:+.2f}%")
c3.metric("Max Drawdown", f"{max_dd:.2f}%")
c4.metric("Trades", len(trades))
c5.metric("Win Rate", f"{win_rate:.1f}%")
c6.metric("Profit Factor", f"{profit_factor:.2f}")

# Grafico de oscilacao de capital (equity curve)
st.subheader("Oscilacao de Capital")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=bt_df.index,
    y=bt_df["Equity"],
    mode="lines",
    name="Equity",
    line=dict(color="#2e7d32", width=2),
    fill="tozeroy",
    fillcolor="rgba(46, 125, 50, 0.1)",
))

# Linha do capital inicial
fig.add_hline(
    y=cfg.initial_capital,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Capital Inicial: ${cfg.initial_capital:,.0f}",
    annotation_position="top left",
)

# Marcar trades no grafico
for t in trades:
    color = "#2e7d32" if t.pnl_pct > 0 else "#c62828"
    marker = "triangle-up" if t.direction == 1 else "triangle-down"
    try:
        ts = pd.Timestamp(t.entry_date)
        idx = bt_df.index.get_indexer([ts], method="nearest")[0]
        eq_val = bt_df["Equity"].iloc[idx]
    except Exception:
        continue

    fig.add_trace(go.Scatter(
        x=[ts],
        y=[eq_val],
        mode="markers",
        marker=dict(symbol=marker, size=8, color=color, opacity=0.6),
        showlegend=False,
        hovertext=f"{t.comment} | P&L: {t.pnl_pct:+.2f}%",
        hoverinfo="text",
    ))

fig.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Data",
    yaxis_title="Capital ($)",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# Ticker flat para comparacao
ALL_TICKERS = {}
for cat, items in ASSETS.items():
    for label, ticker in items.items():
        ALL_TICKERS[label] = ticker

current_ticker = st.session_state.get("bt_symbol", "")
current_yf = None
for label, ticker in ALL_TICKERS.items():
    if label == current_ticker:
        current_yf = ticker
        break

# Comparacao com outros ativos
st.divider()
st.subheader("Correlacao e Distribuicao de Retornos")

compare_options = [l for l in ALL_TICKERS.keys() if l != current_ticker]
selected_compare = st.multiselect(
    "Selecione ativos para comparar:",
    options=compare_options,
    default=compare_options[:3],
)

if selected_compare and current_yf:
    import yfinance as yf

    # Baixar retornos diarios de todos os ativos
    tickers_to_load = {current_ticker: current_yf}
    for label in selected_compare:
        tickers_to_load[label] = ALL_TICKERS[label]

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_returns(tickers_dict):
        closes = {}
        for label, ticker in tickers_dict.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2y", interval="1d")
                if not hist.empty:
                    series = hist["Close"].copy()
                    series.index = series.index.tz_localize(None).normalize()
                    series = series[~series.index.duplicated(keep="first")]
                    closes[label] = series
            except Exception:
                continue
        if not closes:
            return None
        df_closes = pd.DataFrame(closes)
        df_closes = df_closes.ffill().dropna()
        return df_closes.pct_change().dropna()

    with st.spinner("Carregando dados de comparacao..."):
        returns_df = load_returns(tickers_to_load)

    if returns_df is not None and len(returns_df.columns) >= 2:
        tab_corr, tab_hist = st.tabs(["Matriz de Correlacao", "Distribuicao de Retornos"])

        # Correlation Matrix / Heatmap
        with tab_corr:
            corr = returns_df.corr()

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdYlGn",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=14),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlacao: %{z:.3f}<extra></extra>",
            ))
            fig_corr.update_layout(
                title=f"Matriz de Correlacao - Retornos Diarios (2 anos)",
                template="plotly_white",
                height=500,
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.caption(
                "Valores proximos de +1 indicam ativos que se movem juntos. "
                "Valores proximos de -1 indicam movimentos opostos (diversificacao). "
                "Valores proximos de 0 indicam pouca relacao."
            )

        # Histograms & PDF
        with tab_hist:
            # Botoes para filtrar elementos do grafico
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                show_hist = st.toggle("Mostrar Histogramas", value=True)
            with btn_col2:
                show_pdf = st.toggle("Mostrar Curvas PDF", value=True)
            with btn_col3:
                pass

            fig_hist = go.Figure()
            colors = px.colors.qualitative.Set2

            for i, col in enumerate(returns_df.columns):
                ret = returns_df[col].dropna()
                color = colors[i % len(colors)]

                if show_hist:
                    fig_hist.add_trace(go.Histogram(
                        x=ret * 100,
                        name=col,
                        opacity=0.6,
                        nbinsx=50,
                        marker_color=color,
                        histnorm="probability density",
                        hovertemplate=(
                            "<b>%{x:.2f}%</b><br>"
                            "Densidade: %{y:.4f}<br>"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    ))

                if show_pdf:
                    mu = ret.mean() * 100
                    sigma = ret.std() * 100
                    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
                    pdf = stats.norm.pdf(x_range, mu, sigma)
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=pdf,
                        mode="lines",
                        name=f"{col} (Normal)",
                        line=dict(color=color, width=2, dash="dash"),
                        hovertemplate=(
                            "<b>%{x:.2f}%</b><br>"
                            "Densidade teorica: %{y:.4f}<br>"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    ))

            fig_hist.update_layout(
                title="Distribuicao de Retornos Diarios (%)",
                template="plotly_white",
                height=500,
                xaxis_title="Retorno Diario (%)",
                yaxis_title="Densidade",
                barmode="overlay",
                hovermode="x unified",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Tooltips CSS
            st.markdown(
                '<style>'
                '.tt{display:inline-block;position:relative;margin-right:12px}'
                '.tt .tp{visibility:hidden;background:#333;color:#fff;text-align:left;'
                'border-radius:6px;padding:8px 12px;position:absolute;z-index:1;'
                'bottom:130%;left:50%;transform:translateX(-50%);width:320px;'
                'font-size:13px;line-height:1.4;white-space:normal}'
                '.tt:hover .tp{visibility:visible}'
                '.tl{cursor:help;border-bottom:1px dotted #888;font-weight:bold}'
                '</style>',
                unsafe_allow_html=True,
            )

            # Descricoes dos elementos do grafico
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0">'
                '<div class="tt"><span class="tl">Histograma (barras)</span>'
                '<span class="tp">As barras mostram a frequencia real dos retornos diarios. '
                'Barras mais altas indicam retornos que ocorreram mais vezes. '
                'A largura de cada barra representa uma faixa de retorno.</span></div>'
                '<div class="tt"><span class="tl">Curva PDF (linhas)</span>'
                '<span class="tp">A linha tracejada representa a distribuicao normal teorica '
                'com a mesma media e desvio padrao dos retornos reais. '
                'Se as barras se afastam da curva nas extremidades, o ativo tem mais '
                'eventos extremos do que uma distribuicao normal preve (caudas pesadas).</span></div>'
                '<div class="tt"><span class="tl">Densidade</span>'
                '<span class="tp">Eixo Y - representa a probabilidade relativa de um retorno ocorrer. '
                'Quanto mais alto o valor, mais frequente e aquele retorno. '
                'A area total sob a curva soma 1 (100%).</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Tabela de estatisticas
            stats_rows = []
            for col in returns_df.columns:
                ret = returns_df[col].dropna()
                skew_val = ret.skew()
                kurt_val = ret.kurtosis()
                sharpe_val = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0

                skew_desc = "neutra"
                if skew_val < -0.5:
                    skew_desc = "cauda pesada para perdas"
                elif skew_val < -0.1:
                    skew_desc = "leve vies negativo"
                elif skew_val > 0.5:
                    skew_desc = "cauda pesada para ganhos"
                elif skew_val > 0.1:
                    skew_desc = "leve vies positivo"

                kurt_desc = "normal"
                if kurt_val > 3:
                    kurt_desc = "caudas muito pesadas"
                elif kurt_val > 1:
                    kurt_desc = "caudas pesadas"
                elif kurt_val < -0.5:
                    kurt_desc = "caudas leves"

                stats_rows.append({
                    "Ativo": col,
                    "Retorno Medio (%)": f"{ret.mean() * 100:.4f}",
                    "Volatilidade (%)": f"{ret.std() * 100:.4f}",
                    "Skewness": f"{skew_val:.3f}",
                    "Skew Desc": skew_desc,
                    "Kurtosis": f"{kurt_val:.3f}",
                    "Kurt Desc": kurt_desc,
                    "Sharpe (anual)": f"{sharpe_val:.2f}",
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

            # Tooltips das metricas da tabela
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0">'
                '<div class="tt"><span class="tl">Skewness</span>'
                '<span class="tp">Mede a assimetria da distribuicao. '
                '0 = simetrica. '
                'Negativa = cauda mais longa para perdas (perdas extremas mais provaveis). '
                'Positiva = cauda mais longa para ganhos.</span></div>'
                '<div class="tt"><span class="tl">Kurtosis</span>'
                '<span class="tp">Mede o peso das caudas (eventos extremos). '
                '0 = igual a distribuicao normal. '
                'Positiva (leptocurtica) = caudas pesadas, mais eventos extremos. '
                'Negativa (platicurtica) = caudas leves, menos eventos extremos.</span></div>'
                '<div class="tt"><span class="tl">Volatilidade</span>'
                '<span class="tp">Desvio padrao dos retornos diarios. '
                'Quanto maior, mais o preco oscila dia a dia. '
                'Alta volatilidade = mais risco e mais oportunidade.</span></div>'
                '<div class="tt"><span class="tl">Sharpe</span>'
                '<span class="tp">Retorno ajustado pelo risco (anualizado). '
                'Quanto maior, melhor o retorno por unidade de risco. '
                'Acima de 1 e considerado bom, acima de 2 e excelente.</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("Nao foi possivel carregar dados para comparacao.")
elif not current_yf:
    st.info("Comparacao disponivel apenas para ativos da lista pre-definida.")
