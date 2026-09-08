"""
Iniciar: python server.py
"""

import io
import json
import math
import threading
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import stats as scipy_stats

from loader import load_csv
from config import DATA_DIR, TOP_N, COLUMN_DISPLAY
from charts.scatter import (
    plot_return_vs_drawdown,
    plot_return_vs_sharpe,
    plot_return_vs_trades,
)
import itertools

# Diretório com os módulos de estratégia
STRATEGIES_DIR = Path(__file__).parent / "strategies"


def _load_strategy(strategy_file: str):
    """Carrega dinamicamente um módulo de estratégia de strategies/<name>.py."""
    # Segurança: impede path traversal e arquivos privados
    if not strategy_file or strategy_file.startswith("_") \
            or "/" in strategy_file or "\\" in strategy_file \
            or "." in strategy_file:
        raise ValueError(f"Nome de estratégia inválido: '{strategy_file}'")
    path = STRATEGIES_DIR / f"{strategy_file}.py"
    if not path.exists():
        raise FileNotFoundError(f"Estratégia '{strategy_file}' não encontrada em strategies/")
    spec = importlib.util.spec_from_file_location(f"strategies.{strategy_file}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar estrategia '{strategy_file}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])


# Candles a paginar por timeframe no caminho CCXT (~1 ano de 15m, como na
# pesquisa; timeframes maiores pegam mais anos com menos requisições).
_CCXT_TOTALS = {
    "1m": 40000, "5m": 40000, "15m": 35000, "30m": 17600,
    "1h": 17600, "2h": 8800, "4h": 4400, "1d": 3000,
}

# Cache de OHLCV por (symbol, interval, exchange). 35k candles = ~35 requisições
# paginadas; sem cache, backtest + WFA + gráficos re-baixam tudo e estouram o
# rate limit da exchange (Bybit 10006).
_OHLCV_CACHE: dict = {}
_OHLCV_TTL_S = 1800


def _download_data_safe(symbol: str, interval: str, exchange: str | None = None) -> pd.DataFrame:
    """Versão segura de download_data — nunca chama sys.exit.

    Se `exchange` for uma exchange cripto suportada (binance/bybit/okx/
    hyperliquid), busca candles via CCXT (dados públicos, read-only). Caso
    contrário, usa yfinance como antes. O DataFrame de saída tem o mesmo
    formato nos dois caminhos: colunas Open/High/Low/Close/Volume e índice
    DatetimeIndex (UTC, tz-naive).
    """
    if exchange:
        from providers.market_data import SUPPORTED_EXCHANGES, fetch_ohlcv
        if exchange.lower() in SUPPORTED_EXCHANGES:
            import time as _time
            key = (symbol.upper(), interval, exchange.lower())
            hit = _OHLCV_CACHE.get(key)
            if hit and _time.time() - hit[0] < _OHLCV_TTL_S:
                return hit[1].copy()
            # ~1 ano p/ intraday curto (35k de 15m = setup da pesquisa do
            # RELATORIO_prop_challenge); a exchange devolve o que tiver.
            total = _CCXT_TOTALS.get(interval, 5000)
            try:
                df = fetch_ohlcv(symbol, interval, exchange=exchange.lower(), total=total)
            except Exception:
                if hit is not None:      # cache vencido > erro (ex.: rate limit)
                    return hit[1].copy()
                raise
            _OHLCV_CACHE[key] = (_time.time(), df)
            return df.copy()
        raise ValueError(
            f"exchange '{exchange}' não suportada. "
            f"Opções: {', '.join(SUPPORTED_EXCHANGES)}"
        )

    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance não instalado. Execute: pip install yfinance")

    # Mapeia símbolos USDT → yfinance
    yf_symbol = symbol
    if "USDT" in symbol.upper():
        yf_symbol = symbol.upper().replace("USDT", "-USD")

    # yfinance nao suporta 2h e 4h diretamente — baixa 1h e resampla
    # Intervalos intraday tem limite de periodo no yfinance
    resample_rule = None
    yf_interval = interval
    yf_period = "max"

    if interval in ("2h", "4h"):
        yf_interval = "1h"
        yf_period = "2y"
        resample_rule = "2h" if interval == "2h" else "4h"
    elif interval == "15m":
        yf_period = "60d"
    elif interval == "30m":
        yf_period = "60d"
    elif interval == "1h":
        yf_period = "2y"

    ticker = yf.Ticker(yf_symbol)
    # O timeout padrão do curl usado pelo yfinance é curto (10 s). No diário,
    # `period="max"` pode começar a transferir décadas de candles e expirar no
    # meio; o yfinance então devolve um DataFrame vazio, embora o ticker exista.
    df = ticker.history(period=yf_period, interval=yf_interval, timeout=30)

    # Se o histórico máximo ainda falhar (Yahoo instável/rate limit), tenta uma
    # janela menor. Dois anos mantêm ~500 candles — suficientes para os maiores
    # warm-ups das estratégias atuais — e evitam deixar o gráfico sem dados.
    if df.empty and yf_interval == "1d" and yf_period == "max":
        df = ticker.history(period="2y", interval="1d", timeout=30)

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para '{yf_symbol}' (intervalo={yf_interval})")

    # Garante colunas padrão
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Remove timezone do índice
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)  # type: ignore[union-attr]

    # Resampla para 2h ou 4h a partir de 1h
    if resample_rule:
        df = df.resample(resample_rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    return df.sort_index()

# Cache simples para correlações (evita re-download do yfinance)
_corr_cache: dict = {}

# Lista de ativos predefinidos (copiada de pages/backtest_live.py)
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

# Flat map label → ticker para busca rápida
ALL_TICKERS = {
    label: ticker
    for cat in ASSETS.values()
    for label, ticker in cat.items()
}

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

# Mapeamento: chave do CSV (dashboard) -> chave do CONFIG_SCHEMA (otimizador)
DASHBOARD_TO_OPTIMIZER_KEY = {
    "ma": "ma_type",
    "periodo": "ma_length",
    "lookback": "lookback",
    "angulo": "th_up",
    "saida": "exit_mode",
    "banda_pct": "pct_up",
    "alvo_fixo_pct": "alvo_fixo",
    "flat": "exit_on_flat",
    "stop": "stop_type",
    "stop_param": "stop_param",
    "pullback": "use_pullback",
    "entry_zone": "entry_zone",
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _safe(val):
    """Converte NaN/inf para None para serialização JSON segura."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _df_to_records(df: pd.DataFrame) -> list:
    """Converte DataFrame para lista de dicts, tratando NaN → None."""
    df = df.where(pd.notnull(df), other=None)  # type: ignore[arg-type]
    return df.to_dict(orient="records")


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Aplica filtros de trades, sharpe e retorno."""
    result = df.copy()
    min_trades = filters.get("min_trades", 0)
    min_sharpe = filters.get("min_sharpe", None)
    return_min = filters.get("return_min", None)
    return_max = filters.get("return_max", None)

    if "trades" in result.columns and min_trades is not None:
        result = result[result["trades"] >= min_trades]
    if "sharpe" in result.columns and min_sharpe is not None:
        result = result[result["sharpe"] >= min_sharpe]
    if "return_pct" in result.columns:
        if return_min is not None:
            result = result[result["return_pct"] >= return_min]
        if return_max is not None:
            result = result[result["return_pct"] <= return_max]
    return result


def _build_summary(df: pd.DataFrame) -> dict:
    """Calcula métricas resumo do DataFrame filtrado."""
    summary = {"total_params": len(df)}
    if "return_pct" in df.columns and len(df) > 0:
        summary["avg_return"] = _safe(float(df["return_pct"].mean()))  # type: ignore[arg-type]
    else:
        summary["avg_return"] = None
    if "sharpe" in df.columns and len(df) > 0:
        summary["avg_sharpe"] = _safe(float(df["sharpe"].mean()))  # type: ignore[arg-type]
    else:
        summary["avg_sharpe"] = None
    if "score" in df.columns and len(df) > 0:
        summary["best_score"] = _safe(float(df["score"].max()))  # type: ignore[arg-type]
    else:
        summary["best_score"] = None
    return summary


def _build_charts(df: pd.DataFrame) -> dict:
    """Gera os 3 gráficos scatter como JSON Plotly."""
    charts = {}
    if all(c in df.columns for c in ["max_dd_pct", "return_pct", "score"]):
        fig = plot_return_vs_drawdown(df)
        charts["return_vs_drawdown"] = json.loads(fig.to_json())  # type: ignore[arg-type]
    if all(c in df.columns for c in ["sharpe", "return_pct", "win_rate_pct"]):
        fig = plot_return_vs_sharpe(df)
        charts["return_vs_sharpe"] = json.loads(fig.to_json())  # type: ignore[arg-type]
    if all(c in df.columns for c in ["trades", "return_pct", "profit_factor"]):
        fig = plot_return_vs_trades(df)
        charts["return_vs_trades"] = json.loads(fig.to_json())  # type: ignore[arg-type]
    return charts


def _build_table(df: pd.DataFrame, filters: dict) -> list:
    """Retorna top-N linhas ordenadas para a tabela."""
    top_n = filters.get("top_n", TOP_N)
    sort_by = filters.get("sort_by", "score")
    sort_asc = filters.get("sort_asc", False)

    sort_map = {
        "score": "score",
        "return": "return_pct",
        "profit_factor": "profit_factor",
        "sharpe": "sharpe",
        "win_rate": "win_rate_pct",
        "trades": "trades",
        "drawdown_asc": "max_dd_pct",
        "drawdown_desc": "max_dd_pct",
    }
    col = sort_map.get(sort_by, "score")

    if col not in df.columns:
        col = "score" if "score" in df.columns else df.columns[0]

    top = df.sort_values(col, ascending=sort_asc).head(top_n)
    display_cols = [
        "ativo", "rank", "return_pct", "max_dd_pct", "trades", "win_rate_pct",
        "profit_factor", "sharpe", "score", "ma", "periodo", "lookback",
    ]
    available = [c for c in display_cols if c in top.columns]
    return _df_to_records(top[available])


def _build_best_params(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Extrai análise de melhores parâmetros (lógica de best_params.py)."""
    if "score" not in df.columns:
        return {}

    top = df.nlargest(top_n, "score")
    result = {"categorical": {}, "numeric": {}, "summary_table": []}

    # Categóricos
    for param, label in CATEGORICAL_PARAMS.items():
        if param not in top.columns:
            continue
        counts = top[param].value_counts()
        if counts.empty:
            continue
        result["categorical"][param] = {
            "label": label,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "top": str(counts.index[0]),
            "pct": round(float(counts.iloc[0]) / len(top) * 100, 1),
        }
        opt_key = DASHBOARD_TO_OPTIMIZER_KEY.get(param, param)
        top_values = [str(v) for v in counts.head(3).index]
        result["summary_table"].append({
            "Parametro": label,
            "Melhor Valor": str(counts.index[0]),
            "Frequencia no Top": f"{result['categorical'][param]['pct']:.0f}%",
            "Faixa Sugerida": ", ".join(top_values),
            "optimizer_key": opt_key,
            "type": "categorical",
            "values": top_values,
        })

    # Numéricos
    for param, label in NUMERIC_PARAMS.items():
        if param not in top.columns:
            continue
        # Garante tipo numérico
        series = pd.to_numeric(top[param], errors="coerce").dropna()
        if series.empty:
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        median = float(series.median())
        result["numeric"][param] = {
            "label": label,
            "median": _safe(median),
            "q25": _safe(q25),
            "q75": _safe(q75),
            "values": [_safe(v) for v in series.tolist()],
        }
        opt_key = DASHBOARD_TO_OPTIMIZER_KEY.get(param, param)
        result["summary_table"].append({
            "Parametro": label,
            "Melhor Valor": f"{median:.2f} (mediana)",
            "Frequencia no Top": f"{len(series)}/{len(top)}",
            "Faixa Sugerida": f"{q25:.2f} a {q75:.2f}",
            "optimizer_key": opt_key,
            "type": "numeric",
            "min": q25,
            "max": q75,
        })

    return result


def _map_backtest_params(row: dict) -> dict:
    """
    Mapeia campos do CSV para parâmetros do Config (lógica de strategy_detail.py).
    """
    params = {}

    if row.get("ma") is not None:
        params["ma_type"] = str(row["ma"]).upper()

    if row.get("periodo") is not None:
        try:
            params["ma_length"] = int(float(row["periodo"]))
        except (ValueError, TypeError):
            pass

    if row.get("lookback") is not None:
        try:
            params["lookback"] = int(float(row["lookback"]))
        except (ValueError, TypeError):
            pass

    if row.get("angulo") is not None:
        try:
            val = float(row["angulo"])
            params["th_up"] = abs(val)
            params["th_dn"] = -abs(val)
        except (ValueError, TypeError):
            pass

    if row.get("saida") is not None:
        saida = str(row["saida"])
        mode_map = {
            "Banda": "Banda + Tendência",
            "Banda + Tendência": "Banda + Tendência",
            "Tendência": "Somente Tendência",
            "Somente Tendência": "Somente Tendência",
            "Alvo Fixo": "Alvo Fixo + Tendência",
            "Alvo Fixo + Tendência": "Alvo Fixo + Tendência",
        }
        params["exit_mode"] = mode_map.get(saida, saida)

    if row.get("banda_pct") is not None:
        try:
            val = float(row["banda_pct"])
            params["pct_up"] = val
            params["pct_dn"] = val
        except (ValueError, TypeError):
            pass

    if row.get("alvo_fixo_pct") is not None:
        try:
            params["alvo_fixo"] = float(row["alvo_fixo_pct"])
        except (ValueError, TypeError):
            pass

    if row.get("flat") is not None:
        val = row["flat"]
        params["exit_on_flat"] = val in [True, 1, "True", "true", "Sim", "sim", "S", "s", "1"]

    if row.get("stop") is not None:
        stop_val = str(row["stop"])
        if stop_val.lower() in ["off", "false", "0", "nao", "não", ""]:
            params["use_stop"] = False
        else:
            params["use_stop"] = True
            params["stop_type"] = stop_val

    if row.get("stop_param") is not None:
        try:
            v = float(row["stop_param"])
            params["stop_atr_mult"] = v
            params["stop_fixo_pct"] = v
            params["stop_band_pct"] = v
        except (ValueError, TypeError):
            pass

    if row.get("pullback") is not None:
        val = row["pullback"]
        params["use_pullback"] = val in [True, 1, "True", "true", "Sim", "sim", "S", "s", "1"]

    if row.get("entry_zone") is not None:
        val = row["entry_zone"]
        params["use_entry_zone"] = val in [True, 1, "True", "true", "Sim", "sim", "S", "s", "1"]

    return params


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/files", methods=["GET"])
def api_files():
    """Lista arquivos CSV na pasta data/."""
    try:
        if not DATA_DIR.exists():
            return jsonify({"files": []})
        files = sorted([f.name for f in DATA_DIR.glob("*.csv")])
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load", methods=["POST"])
def api_load():
    """
    Carrega CSV (upload ou filename) e retorna dados brutos + filtros padrão + charts.
    Aceita multipart/form-data (file) ou application/json (filename + filters).
    """
    try:
        filters = {}
        df = None
        filename = ""

        if "file" in request.files:
            file_obj = request.files["file"]
            filename = file_obj.filename or "upload.csv"
            df = load_csv(file_obj)
            # Filtros do form
            filters_raw = request.form.get("filters", "{}")
            filters = json.loads(filters_raw)
        else:
            body = request.get_json(force=True) or {}
            filename = body.get("filename", "")
            filters = body.get("filters", {})
            if not filename:
                return jsonify({"error": "filename ou file obrigatório"}), 400
            path = DATA_DIR / filename
            if not path.exists():
                return jsonify({"error": f"Arquivo não encontrado: {filename}"}), 404
            df = load_csv(path)

        # Ranges reais do CSV completo
        filter_ranges = {}
        if "sharpe" in df.columns:
            filter_ranges["sharpe_min"] = _safe(float(df["sharpe"].min()))
            filter_ranges["sharpe_max"] = _safe(float(df["sharpe"].max()))
        if "return_pct" in df.columns:
            filter_ranges["return_min"] = _safe(float(df["return_pct"].min()))
            filter_ranges["return_max"] = _safe(float(df["return_pct"].max()))
        if "trades" in df.columns:
            trades_clean = df["trades"].dropna()
            if len(trades_clean) > 0:
                filter_ranges["trades_min"] = int(trades_clean.min())
                filter_ranges["trades_max"] = int(trades_clean.max())

        # Aplica filtros
        if not filters:
            filters = {
                "min_trades": 0,
                "min_sharpe": filter_ranges.get("sharpe_min"),
                "return_min": filter_ranges.get("return_min"),
                "return_max": filter_ranges.get("return_max"),
                "top_n": TOP_N,
                "sort_by": "score",
                "sort_asc": False,
            }
        filtered = _apply_filters(df, filters)

        asset = str(df["ativo"].iloc[0]) if "ativo" in df.columns else filename

        return jsonify({
            "asset": asset,
            "filename": filename,
            "total_rows": len(df),
            "filtered_rows": len(filtered),
            "filter_ranges": filter_ranges,
            "raw_rows": _df_to_records(df),
            "summary": _build_summary(filtered),
            "charts": _build_charts(filtered),
            "table": _build_table(filtered, filters),
            "best_params": _build_best_params(filtered, filters.get("top_n", TOP_N)),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/filter", methods=["POST"])
def api_filter():
    """
    Reaplicar filtros sobre os dados brutos (enviados pelo frontend).
    Backend stateless: frontend envia rawRows a cada filtragem.
    """
    try:
        body = request.get_json(force=True) or {}
        rows = body.get("rows", [])
        filters = body.get("filters", {})

        if not rows:
            return jsonify({"error": "rows obrigatório"}), 400

        df = pd.DataFrame(rows)

        # Garante tipos numéricos
        from config import NUMERIC_COLUMNS
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # periodo pode ser numérico para best_params
        if "periodo" in df.columns:
            df["periodo"] = pd.to_numeric(df["periodo"], errors="coerce")

        filtered = _apply_filters(df, filters)

        return jsonify({
            "filtered_rows": len(filtered),
            "summary": _build_summary(filtered),
            "charts": _build_charts(filtered),
            "table": _build_table(filtered, filters),
            "best_params": _build_best_params(filtered, filters.get("top_n", TOP_N)),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/filter-chart", methods=["POST"])
def api_filter_chart():
    """
    Filtra os dados brutos e regenera um unico grafico scatter.
    Recebe: rows, chart_type, chart_filters
    """
    try:
        body = request.get_json(force=True) or {}
        rows = body.get("rows", [])
        chart_type = body.get("chart_type", "")
        chart_filters = body.get("chart_filters", {})

        if not rows:
            return jsonify({"error": "rows obrigatorio"}), 400

        df = pd.DataFrame(rows)
        from config import NUMERIC_COLUMNS
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Aplica filtros visuais do grafico
        if "return_min" in chart_filters and chart_filters["return_min"] is not None:
            df = df[df["return_pct"] >= chart_filters["return_min"]]
        if "dd_max" in chart_filters and chart_filters["dd_max"] is not None:
            df = df[df["max_dd_pct"] >= chart_filters["dd_max"]]
        if "sharpe_min" in chart_filters and chart_filters["sharpe_min"] is not None:
            df = df[df["sharpe"] >= chart_filters["sharpe_min"]]
        if "trades_min" in chart_filters and chart_filters["trades_min"] is not None:
            df = df[df["trades"] >= chart_filters["trades_min"]]

        chart_json = None
        if chart_type == "return_vs_drawdown" and all(c in df.columns for c in ["max_dd_pct", "return_pct", "score"]):
            chart_json = json.loads(plot_return_vs_drawdown(df).to_json())  # type: ignore[arg-type]
        elif chart_type == "return_vs_sharpe" and all(c in df.columns for c in ["sharpe", "return_pct", "win_rate_pct"]):
            chart_json = json.loads(plot_return_vs_sharpe(df).to_json())  # type: ignore[arg-type]
        elif chart_type == "return_vs_trades" and all(c in df.columns for c in ["trades", "return_pct", "profit_factor"]):
            chart_json = json.loads(plot_return_vs_trades(df).to_json())  # type: ignore[arg-type]

        return jsonify({
            "chart": chart_json,
            "count": len(df),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategy", methods=["POST"])
def api_strategy():
    """
    Retorna detalhes de uma estratégia por rank + parâmetros mapeados para backtest.
    """
    try:
        body = request.get_json(force=True) or {}
        rank = body.get("rank")
        rows = body.get("rows", [])

        if not rows:
            return jsonify({"error": "rows obrigatório"}), 400

        df = pd.DataFrame(rows)
        from config import NUMERIC_COLUMNS
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if rank is not None and "rank" in df.columns:
            match = df[df["rank"] == rank]
            if match.empty:
                # Tenta por índice como fallback
                try:
                    row_dict = df.iloc[int(rank)].to_dict()
                except Exception:
                    return jsonify({"error": f"rank {rank} não encontrado"}), 404
            else:
                row_dict = match.iloc[0].to_dict()
        else:
            return jsonify({"error": "rank obrigatório"}), 400

        # Limpa NaN
        detail = {k: (_safe(v) if isinstance(v, float) else v) for k, v in row_dict.items()}
        backtest_params = _map_backtest_params(row_dict)

        return jsonify({"detail": detail, "backtest_params": backtest_params})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/assets", methods=["GET"])
def api_backtest_assets():
    """Retorna dicionário de ativos predefinidos."""
    return jsonify({"assets": ASSETS})


@app.route("/api/backtest/exchanges", methods=["GET"])
def api_backtest_exchanges():
    """Exchanges cripto suportadas via CCXT (dados públicos OHLCV).

    Quando o frontend manda `exchange` no /api/backtest/run (e afins), os
    candles vêm da exchange escolhida em vez do yfinance. `symbol` pode ser só
    a base (ex.: 'BTC') — é normalizado pro perp da exchange automaticamente.
    """
    from providers.market_data import SUPPORTED_EXCHANGES
    return jsonify({"exchanges": list(SUPPORTED_EXCHANGES)})


@app.route("/api/backtest/strategies", methods=["GET"])
def api_backtest_strategies():
    """Lista todos os módulos de estratégia disponíveis em strategies/."""
    strategies = []
    if STRATEGIES_DIR.exists():
        for path in sorted(STRATEGIES_DIR.glob("*.py")):
            if path.name.startswith("_"):
                continue  # ignora __init__.py e _utils.py
            try:
                module = _load_strategy(path.stem)
                strategies.append({
                    "file": path.stem,
                    "name": getattr(module, "NAME", path.stem),
                    "description": getattr(module, "DESCRIPTION", ""),
                    "schema": getattr(module, "CONFIG_SCHEMA", []),
                    # automatizável = expõe signal() (contrato do módulo automation)
                    "automatable": callable(getattr(module, "signal", None)),
                })
            except Exception as e:
                import traceback
                strategies.append({
                    "file": path.stem,
                    "name": path.stem,
                    "description": "",
                    "schema": [],
                    "error": str(e),
                })
    return jsonify({"strategies": strategies})


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    """
    Executa o backtest delegando ao módulo de estratégia escolhido.
    Aceita:
    - JSON: { "strategy_file": "depaula", "data_source": "asset",
              "symbol": "BTC-USD", "interval": "1d", "config": {...} }
    - multipart/form-data: file CSV + config (JSON) + strategy_file (form field)
    """
    try:
        df_data = None
        cfg_dict = {}
        symbol_label = ""
        interval_label = ""
        strategy_file = "depaula"  # padrão para backward compat

        if "file" in request.files:
            file_obj = request.files["file"]
            symbol_label = file_obj.filename or "CSV"
            interval_label = "-"
            cfg_raw = request.form.get("config", "{}")
            cfg_dict = json.loads(cfg_raw)
            strategy_file = request.form.get("strategy_file", "depaula") or "depaula"
            # Lê CSV de preços (colunas: Date,Open,High,Low,Close)
            raw = file_obj.read()
            for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
                try:
                    text = raw.decode(enc)
                    df_data = pd.read_csv(
                        io.StringIO(text),
                        parse_dates=["Date"],
                        index_col="Date"
                    ).sort_index()
                    break
                except Exception:
                    continue
            if df_data is None:
                return jsonify({"error": "Não foi possível ler o CSV de preços"}), 400
        else:
            body = request.get_json(force=True) or {}
            cfg_dict = body.get("config", {})
            strategy_file = body.get("strategy_file", "depaula") or "depaula"
            symbol_label = body.get("symbol_label", "")
            interval_label = body.get("interval", "1d")
            symbol = body.get("symbol", "")
            exchange = body.get("exchange")

            if not symbol:
                return jsonify({"error": "symbol obrigatório"}), 400

            df_data = _download_data_safe(symbol, interval_label, exchange)
            if symbol_label == "":
                symbol_label = symbol

        # Carrega o módulo de estratégia e executa
        module = _load_strategy(strategy_file)
        result_dict = module.run(df_data.copy(), cfg_dict)

        return jsonify({
            "symbol": symbol_label,
            "interval": interval_label,
            "strategy": strategy_file,
            **result_dict,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/backtest/chart-data", methods=["POST"])
def api_backtest_chart_data():
    """Dados para os gráficos de análise (Plotly) de um backtest.

    Roda a estratégia (com `_charts`) e devolve, para o mesmo símbolo/intervalo:
      - candles   : OHLCV (do próprio backtest)
      - indicators: MA / banda superior / banda inferior
      - trades    : marcadores de entrada/saída (long/short)
      - equity    : curva bruta vs líquida (descontando fees + funding reais)
      - funding   : histórico de funding rate da corretora escolhida

    Body: igual ao /api/backtest/run + cost_exchange/cost_scenario/use_funding/
    cost_symbol (opcionais; default = exchange dos dados ou 'binance').
    """
    try:
        body = request.get_json(force=True) or {}
        cfg_dict = dict(body.get("config", {}))
        strategy_file = body.get("strategy_file", "depaula") or "depaula"
        interval_label = body.get("interval", "1d")
        symbol = body.get("symbol", "")
        exchange = body.get("exchange")

        if not symbol:
            return jsonify({"error": "symbol obrigatório"}), 400

        df_data = _download_data_safe(symbol, interval_label, exchange)
        module = _load_strategy(strategy_file)
        result = module.run(df_data.copy(), {**cfg_dict, "_charts": True})

        chart = result.get("chart") or {}
        raw_trades = result.get("trades", [])

        # Marcadores de trade (long/short) com data/preço de entrada e saída.
        markers = []
        for t in raw_trades:
            if t.get("entry_ts") and t.get("entry_price") is not None:
                markers.append({
                    "entry_ts":    t.get("entry_ts"),
                    "exit_ts":     t.get("exit_ts"),
                    "entry_price": t.get("entry_price"),
                    "exit_price":  t.get("exit_price"),
                    "direction":   t.get("direction", 1),
                    "pnl_pct":     t.get("pnl_pct"),
                    "stop_price":   t.get("stop_price"),
                    "target_price": t.get("target_price"),
                })

        # Contexto de custos: força apply_costs p/ sempre montar funding + líquido.
        cost_ctx, cost_warnings = _build_wfa_cost_ctx(
            df_data, {**body, "apply_costs": body.get("apply_costs", True)}, cfg_dict
        )

        # Curva de equity bruta (por barra) + líquida (bruta − custo acumulado).
        eq_dates = result.get("equity_curve", {}).get("dates", [])
        eq_gross = result.get("equity_curve", {}).get("values", [])
        eq_net   = None
        funding  = {"dates": [], "rates": []}

        if cost_ctx is not None:
            from costs import trades_from_platform
            calc = cost_ctx["calc"]
            funding_events = cost_ctx["funding_events"]

            # Custo absoluto por trade (positivo), associado ao timestamp de saída.
            costs_by_exit = []
            for ct in trades_from_platform(raw_trades):
                bd = calc.apply_trade(ct, funding_events)
                costs_by_exit.append((ct.exit_time, float(bd.pnl_bruto - bd.pnl_liquido)))
            costs_by_exit.sort(key=lambda x: x[0])

            # Alinha custo acumulado às barras pelo timestamp completo do chart.
            bar_ts = []
            for d in (chart.get("dates") or []):
                try:
                    bar_ts.append(int(pd.Timestamp(d).value // 1_000_000))
                except Exception:
                    bar_ts.append(None)

            if bar_ts and len(bar_ts) == len(eq_gross):
                eq_net = []
                ci, running = 0, 0.0
                for i, bts in enumerate(bar_ts):
                    while ci < len(costs_by_exit) and bts is not None and costs_by_exit[ci][0] <= bts:
                        running += costs_by_exit[ci][1]
                        ci += 1
                    g = eq_gross[i]
                    eq_net.append(_safe(g - running) if g is not None else None)

            funding = {
                "dates": [pd.Timestamp(ev.timestamp, unit="ms").isoformat() for ev in funding_events],
                "rates": [float(ev.rate) for ev in funding_events],
            }

        return jsonify({
            "symbol":     body.get("symbol_label") or symbol,
            "interval":   interval_label,
            "candles":    {"dates": chart.get("dates", []), **(chart.get("ohlc") or {})},
            "indicators": chart.get("indicators", {}),
            "trades":     markers,
            "equity":     {"dates": eq_dates, "gross": eq_gross, "net": eq_net},
            "funding":    funding,
            "cost_exchange": cost_ctx["exchange"] if cost_ctx else None,
            "cost_scenario": cost_ctx["scenario"] if cost_ctx else None,
            "cost_warnings": cost_warnings,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/backtest/costs", methods=["POST"])
def api_backtest_costs():
    """
    Aplica fees + funding rate aos trades de um backtest e devolve a comparação
    bruto vs líquido por exchange (Binance / Bybit / OKX) e por cenário.

    Body: {
      "trades": [...],                  # trades emitidos por strategies/*.run()
      "symbol": "BTC/USDT:USDT",        # símbolo CCXT (swap)
      "exchanges": ["binance","bybit","okx"],
      "scenarios": ["realista","pessimista"],
      "initial_capital": 1000.0,
      "use_funding": true,              # se false, custeia só fees (sem rede)
      "fees": {"binance":{"maker":..,"taker":..}, ...}   # override opcional de tier
    }
    """
    try:
        from decimal import Decimal as _Dec
        from costs import trades_from_platform, compare_exchanges, DEFAULT_FEES
        from costs.config import ExchangeFees as _Fees
        from costs.funding import get_funding_events as _get_funding
        import costs.compare as _ccompare
    except Exception as e:
        return jsonify({"error": f"Módulo de custos indisponível: {e}"}), 500

    try:
        body = request.get_json(force=True) or {}
        raw_trades = body.get("trades", [])
        symbol = body.get("symbol") or "BTC/USDT:USDT"
        exchanges = tuple(body.get("exchanges") or ("binance", "bybit", "okx"))
        scenarios = body.get("scenarios") or ["realista", "pessimista"]
        initial_capital = float(body.get("initial_capital", 1000.0))
        use_funding = bool(body.get("use_funding", True))
        fees_override = body.get("fees") or {}

        trades = trades_from_platform(raw_trades)
        if len(trades) < 1:
            return jsonify({"error": "Nenhum trade com qty/timestamp para custear. "
                                     "Rode o backtest com sizing (qty/alavancagem) definido."}), 400

        # Override de fees por tier (mantém o default se não informado)
        for ex, f in fees_override.items():
            if ex in DEFAULT_FEES and "maker" in f and "taker" in f:
                DEFAULT_FEES[ex] = _Fees(maker=_Dec(str(f["maker"])),
                                         taker=_Dec(str(f["taker"])))

        warnings = []

        def provider(exchange, sym, since_ms, until_ms):
            if not use_funding:
                return []
            try:
                return _get_funding(exchange, sym, since_ms, until_ms)
            except Exception as ex:
                warnings.append(f"{exchange}: funding indisponível ({ex}). Usando só fees.")
                return []

        frames = []
        for sc in scenarios:
            df_sc = compare_exchanges({body.get("strategy_name", "Estratégia"): trades},
                                      symbol, exchanges=exchanges, scenario=sc,
                                      initial_capital=initial_capital,
                                      funding_provider=provider)
            frames.append(df_sc)

        full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows = [{k: _safe(v) if isinstance(v, (int, float)) else v for k, v in r.items()}
                for r in full.to_dict(orient="records")]

        return jsonify({
            "symbol": symbol,
            "rows": rows,
            "warnings": warnings,
            "n_trades": len(trades),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def _extract_param_specs(module):
    """Extrai params para otimizacao IS a partir de OPTIMIZER_GRIDS['rapido'].
    Retorna (specs_dict, numeric_keys_list).
    """
    grids = getattr(module, 'OPTIMIZER_GRIDS', {})
    grid  = grids.get('rapido', grids.get('default', {}))
    if not grid:
        return {}, []
    specs = {}
    numeric_keys = []
    for key, values in grid.items():
        if not values or len(values) < 2:
            continue
        specs[key] = list(values)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            numeric_keys.append(key)
    return specs, numeric_keys


def _random_config_from_grid(base_config, param_specs, rng):
    """Gera uma config aleatoria a partir das listas de valores do grid."""
    cfg = dict(base_config)
    for key, values in param_specs.items():
        cfg[key] = values[int(rng.integers(0, len(values)))]
    # Mantém simetria de params derivados
    if 'th_up' in cfg:
        cfg['th_dn'] = -abs(float(cfg['th_up']))
    if 'pct_up' in cfg:
        cfg['pct_dn'] = cfg['pct_up']
    return cfg


# Barras por ano por timeframe — usado para anualizar Sharpe corretamente
# FALLBACK de barras/ano (calendario de PREGAO: 6,5h x 252d — nao vale p/
# cripto 24/7). Usado so quando nao da p/ inferir do span real dos dados;
# a WFA e o Monte Carlo inferem de len(barras)/anos do proprio slice.
_BARS_PER_YEAR = {
    '1m': 98280, '5m': 19656, '15m': 6552, '30m': 3276,
    '1h': 1638,  '2h': 819,   '4h': 410,
    '1d': 252,   '1wk': 52,   '1mo': 12,
}


def _build_wfa_cost_ctx(df, body, cfg_dict):
    """Monta o contexto de custos (fees + funding) para a WFA, ou (None, []).

    Quando `apply_costs` está ligado, instancia o CostCalculator da exchange
    escolhida (taxas reais por tier + cenário) e baixa o funding histórico UMA
    vez para todo o range do DataFrame (reaproveitado em todas as janelas).
    O funding é opcional: se a exchange estiver indisponível, cai para só-fees
    e registra um aviso. Retorna (ctx_dict | None, lista_de_avisos).
    """
    if not body.get("apply_costs"):
        return None, []

    from decimal import Decimal as _Dec
    from costs.calculator import CostCalculator
    from costs.config import DEFAULT_FEES, SCENARIOS, ExchangeFees
    from costs.funding import get_funding_events
    from providers.market_data import normalize_symbol

    cost_exchange = (body.get("cost_exchange") or body.get("exchange") or "binance").lower()
    if cost_exchange not in DEFAULT_FEES:
        cost_exchange = "binance"
    scenario = body.get("cost_scenario") or "realista"
    if scenario not in SCENARIOS:
        scenario = "realista"
    use_funding = bool(body.get("use_funding", True))
    initial_capital = float(
        body.get("initial_capital")
        or cfg_dict.get("initial_capital")
        or 1000.0
    )

    # Taxa base do tier + override opcional vindo do frontend.
    fees = DEFAULT_FEES[cost_exchange]
    fo = (body.get("fees") or {}).get(cost_exchange)
    if fo and "maker" in fo and "taker" in fo:
        fees = ExchangeFees(maker=_Dec(str(fo["maker"])), taker=_Dec(str(fo["taker"])))

    # Regra crítica #2: default = TAKER nas duas pontas. Maker só quando o
    # usuário declara explicitamente que a estratégia usa ordens limit no book.
    use_maker_entry = bool(body.get("use_maker_entry", False))
    use_maker_exit = bool(body.get("use_maker_exit", False))

    sc = SCENARIOS[scenario]
    calc = CostCalculator(
        exchange=cost_exchange, fees=fees,
        use_maker_entry=use_maker_entry, use_maker_exit=use_maker_exit,
        funding_multiplier=sc["funding_multiplier"],
        slippage_pct=sc["slippage_pct"],
    )

    warnings = []
    funding_events = []
    if use_funding:
        sym = normalize_symbol(body.get("cost_symbol") or body.get("symbol") or "BTC", cost_exchange)
        since_ms = int(df.index[0].value // 1_000_000)
        until_ms = int(df.index[-1].value // 1_000_000)
        try:
            funding_events = get_funding_events(cost_exchange, sym, since_ms, until_ms)
        except Exception as ex:
            warnings.append(f"{cost_exchange}: funding indisponível ({ex}). Aplicando só fees.")
            funding_events = []

    ctx = {
        "calc":            calc,
        "funding_events":  funding_events,
        "initial_capital": initial_capital,
        "exchange":        cost_exchange,
        "scenario":        scenario,
        "use_funding":     use_funding,
        "use_maker_entry": use_maker_entry,
        "use_maker_exit":  use_maker_exit,
    }
    return ctx, warnings


def _net_metrics_from_result(eval_trades, cost_ctx, base_return_pct, base_equity=None):
    """Aplica fees + funding aos trades de uma janela e devolve métricas líquidas.

    `eval_trades` são os dicts de trade da janela AVALIADA (trades iniciados no
    warm-up ficam de fora). `base_return_pct` é o retorno bruto medido na janela;
    subtraímos dele o arrasto de custo (fees+funding sobre o notional) para obter
    o retorno líquido. `base_equity` é o equity na fronteira da janela — mesma
    base do retorno bruto e do sizing dos trades (fallback: capital inicial).
    """
    from costs import trades_from_platform

    trades = trades_from_platform(eval_trades)
    if not trades:
        return {
            "net_return_pct": None, "net_sharpe": None,
            "fees_total": 0.0, "funding_total": 0.0, "cost_drag_pct": 0.0,
        }

    base = float(base_equity or cost_ctx["initial_capital"] or 1.0)
    res = cost_ctx["calc"].apply(
        trades, cost_ctx["funding_events"],
        initial_capital=base,
    )
    cost_drag_pct = float(res.pnl_liquido_total - res.pnl_bruto_total) / base * 100.0
    return {
        "net_return_pct": _safe(float(base_return_pct or 0.0) + cost_drag_pct),
        "net_sharpe":     _safe(res.metrics_net.get("sharpe")),
        "fees_total":     _safe(float(res.fees_total)),
        "funding_total":  _safe(float(res.funding_total)),
        "cost_drag_pct":  _safe(cost_drag_pct),
    }


def _net_pnl_pcts(raw_trades, cost_ctx):
    """Converte o pool de pnl_pct (bruto, % do equity) em pnl_pct líquido,
    descontando fees + funding reais da corretora trade a trade.

    O `pnl_pct` da plataforma é o retorno do trade sobre o equity no momento da
    entrada (já escalado pela exposição/alavancagem). Para descontar o custo na
    mesma base, reconstruímos o equity de entrada compondo o pnl_pct bruto a
    partir do capital inicial (mesma trajetória do backtest) e dividimos o custo
    absoluto do trade por esse equity. Trades sem dados para custear passam
    intactos. Retorna dict com o pool líquido e os totais de fees/funding.
    """
    from costs import trades_from_platform

    calc = cost_ctx["calc"]
    funding_events = cost_ctx["funding_events"]
    equity = cost_ctx["initial_capital"] or 1.0

    net_pcts = []
    total_fees = 0.0
    total_funding = 0.0
    n_costed = 0

    for t in raw_trades:
        g = t.get("pnl_pct")
        if g is None:
            continue
        g = float(g)
        eq_entry = equity if equity else 1.0
        one = trades_from_platform([t])
        if one:
            bd = calc.apply_trade(one[0], funding_events)
            cost_abs = float(bd.pnl_liquido - bd.pnl_bruto)   # ≤ 0 (fees) ± funding
            net_pcts.append(g + cost_abs / eq_entry * 100.0)
            total_fees += float(bd.fee_entrada + bd.fee_saida)
            total_funding += float(bd.funding_total)
            n_costed += 1
        else:
            net_pcts.append(g)
        equity *= (1 + g / 100.0)   # compõe com o bruto: mesma trajetória do backtest

    return {
        "net_pcts": net_pcts,
        "total_fees": total_fees,
        "total_funding": total_funding,
        "n_costed": n_costed,
    }


def _compute_window_metrics(df_slice, module, cfg_dict, cost_ctx=None, eval_start=0):
    """Roda a estrategia em um slice do DataFrame e mede APENAS a janela avaliada.

    `df_slice` pode incluir um prefixo de warm-up (barras ANTERIORES a janela)
    para aquecer os indicadores como no trading real — sem ele, cada janela OOS
    partia fria e a MA/regime consumia boa parte da janela sem gerar sinais.
    `eval_start` marca onde a avaliacao comeca: retorno, Sharpe, max DD e curva
    de equity sao medidos da fronteira em diante, e so contam trades INICIADOS
    dentro da janela avaliada. (Trade aberto no warm-up que fecha dentro da
    janela entra na equity, mas nao na contagem nem nos custos — impreciso e
    conservador.) Retorna None se a janela for pequena ou gerar <2 trades.

    Se `cost_ctx` for dado, anexa metricas liquidas (fees + funding reais da
    exchange): net_return_pct, net_sharpe, fees_total, funding_total,
    cost_drag_pct.
    """
    if len(df_slice) - eval_start < 20:
        return None
    try:
        result = module.run(df_slice.copy(), {**cfg_dict, "_fast": True})
    except Exception:
        return None

    eq_values = result.get("equity_curve", {}).get("values", [])
    eq_dates  = result.get("equity_curve", {}).get("dates", [])

    # Fronteira da janela avaliada na curva de equity. Alinha pelo FIM: se a
    # estrategia descartar barras iniciais (indicadores NaN), o rabo da curva
    # continua correspondendo as ultimas barras do slice.
    n_eval_bars = len(df_slice) - eval_start
    b = max(len(eq_values) - n_eval_bars, 0)
    if len(eq_values) - b < 3:
        return None
    eq = np.array(eq_values[b:], dtype=float)
    base = eq[0] if eq[0] else 1.0

    # So trades iniciados dentro da janela avaliada (entry_ts em epoch ms).
    # Trades sem timestamp passam (não da p/ situa-los; comportamento antigo).
    boundary_ts = int(df_slice.index[eval_start].value // 1_000_000)
    eval_trades = [t for t in result.get("trades", [])
                   if not t.get("entry_ts") or int(t["entry_ts"]) >= boundary_ts]
    n_trades = len(eval_trades)
    if n_trades < 2:
        return None

    return_pct = (eq[-1] / base - 1) * 100

    # Barras/ano inferidas do proprio slice (24/7 p/ cripto, pregao p/ acoes).
    # A tabela fixa antiga assumia calendario de bolsa (6,5h x 252d) e
    # subestimava o Sharpe de cripto intraday em ~2,3x.
    span_s = (df_slice.index[-1] - df_slice.index[eval_start]).total_seconds()
    years = span_s / (365.25 * 24 * 3600)
    ann_factor = (len(eq) / years) if years > 0 else 252.0

    # Sharpe anualizado da janela avaliada (ddof=1 = variancia amostral)
    sharpe = 0.0
    rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
    rets = rets[np.isfinite(rets)]
    if len(rets) > 1:
        std = rets.std(ddof=1)
        if std > 0:
            sharpe = float(rets.mean() / std * np.sqrt(ann_factor))

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak != 0, peak, 1.0) * 100
    max_dd = float(dd.min()) if len(dd) else 0.0

    out = {
        "return_pct":    _safe(float(return_pct)),
        "sharpe":        _safe(sharpe),
        "n_trades":      n_trades,
        "max_dd":        _safe(max_dd),
        "equity_values": [_safe(float(v)) for v in eq],
        "equity_dates":  eq_dates[b:],
    }

    if cost_ctx is not None:
        out.update(_net_metrics_from_result(
            eval_trades, cost_ctx, return_pct, base_equity=float(base)
        ))

    return out


@app.route("/api/backtest/wfa", methods=["POST"])
def api_backtest_wfa():
    """
    Walk-Forward Analysis.
    Input JSON: { symbol, interval, strategy_file, config, n_windows, is_pct }
    """
    try:
        body = request.get_json(force=True) or {}
        symbol        = body.get("symbol", "")
        interval      = body.get("interval", "1d")
        strategy_file = body.get("strategy_file", "depaula") or "depaula"
        cfg_dict      = body.get("config", {})
        n_windows            = int(max(3, min(30, body.get("n_windows", 10))))
        is_pct               = float(max(0.5, min(0.85, body.get("is_pct", 0.70))))
        optimize_is_samples  = int(max(0, min(200, body.get("optimize_is_samples", 0))))

        if not symbol:
            return jsonify({"error": "symbol obrigatorio para WFA"}), 400

        df = _download_data_safe(symbol, interval, body.get("exchange"))
        total_bars = len(df)

        if total_bars < n_windows * 20:
            return jsonify({
                "error": f"Dados insuficientes ({total_bars} barras) para {n_windows} janelas. "
                         f"Reduza o numero de janelas ou use um timeframe maior."
            }), 400

        module = _load_strategy(strategy_file)
        param_specs, numeric_keys = (
            _extract_param_specs(module) if optimize_is_samples > 0 else ({}, [])
        )

        # Contexto de custos (fees + funding reais da exchange). Funding baixado
        # uma vez para todo o range e reusado em cada janela.
        cost_ctx, cost_warnings = _build_wfa_cost_ctx(df, body, cfg_dict)

        step_size = total_bars // n_windows
        is_bars   = int(step_size * is_pct)
        oos_bars  = step_size - is_bars

        # Warm-up: barras ANTERIORES a cada janela para aquecer indicadores
        # (MA/regime) como no trading real. Sem isso cada janela partia fria e
        # os primeiros ~ma_length candles nao geravam sinais — em janelas OOS
        # curtas isso consumia boa parte da janela. A avaliacao (retorno,
        # Sharpe, trades, custos) continua estritamente dentro da janela.
        # 400 cobre o rank de volatilidade da MM9 pullback (rolling 384).
        warmup_bars = 400

        windows = []
        for i in range(n_windows):
            is_start  = i * step_size
            is_end    = is_start + is_bars
            oos_start = is_end
            # Ultima janela absorve o resto da divisao — sem descartar as
            # barras mais recentes.
            oos_end   = total_bars if i == n_windows - 1 else min(oos_start + oos_bars, total_bars)

            if oos_end <= oos_start:
                continue

            is_warm  = min(is_start, warmup_bars)
            oos_warm = min(oos_start, warmup_bars)
            df_is  = df.iloc[is_start - is_warm:is_end]
            df_oos = df.iloc[oos_start - oos_warm:oos_end]

            # IS optimization: random-sample the grid, keep best Sharpe config.
            # Com custos ligados, ranqueia pelo Sharpe LIQUIDO — o bruto
            # favorece configs que giram demais e morrem depois das fees.
            if optimize_is_samples > 0 and param_specs:
                best_sharpe = float('-inf')
                window_cfg  = dict(cfg_dict)
                win_rng     = np.random.default_rng(42 + i)
                for _ in range(optimize_is_samples):
                    trial_cfg = _random_config_from_grid(cfg_dict, param_specs, win_rng)
                    trial_m   = _compute_window_metrics(df_is, module, trial_cfg,
                                                        cost_ctx, eval_start=is_warm)
                    if trial_m is None:
                        continue
                    if cost_ctx is not None and trial_m.get('net_sharpe') is not None:
                        trial_sharpe = trial_m['net_sharpe']
                    else:
                        trial_sharpe = trial_m['sharpe'] if trial_m['sharpe'] is not None else 0.0
                    if trial_sharpe > best_sharpe:
                        best_sharpe = trial_sharpe
                        window_cfg  = trial_cfg
            else:
                window_cfg = dict(cfg_dict)

            is_m  = _compute_window_metrics(df_is,  module, window_cfg, cost_ctx, eval_start=is_warm)
            oos_m = _compute_window_metrics(df_oos, module, window_cfg, cost_ctx, eval_start=oos_warm)

            if is_m is None or oos_m is None:
                continue

            # Annualized return: (1 + r)^(365/days) - 1  (janela avaliada, sem warm-up)
            is_days  = max((df.index[is_end - 1]  - df.index[is_start]).days,  1)
            oos_days = max((df.index[oos_end - 1] - df.index[oos_start]).days, 1)
            is_ann  = _safe(((1 + (is_m["return_pct"]  or 0) / 100) ** (365 / is_days)  - 1) * 100)
            oos_ann = _safe(((1 + (oos_m["return_pct"] or 0) / 100) ** (365 / oos_days) - 1) * 100)

            windows.append({
                "window_idx":       i,
                "is_start":         str(df.index[is_start])[:10],
                "is_end":           str(df.index[is_end - 1])[:10],
                "oos_start":        str(df.index[oos_start])[:10],
                "oos_end":          str(df.index[oos_end - 1])[:10],
                "is_return":        is_m["return_pct"],
                "oos_return":       oos_m["return_pct"],
                "is_annualized":    is_ann,
                "oos_annualized":   oos_ann,
                "is_sharpe":        is_m["sharpe"],
                "oos_sharpe":       oos_m["sharpe"],
                "is_trades":        is_m["n_trades"],
                "oos_trades":       oos_m["n_trades"],
                "is_equity":        is_m["equity_values"],
                "oos_equity":       oos_m["equity_values"],
                "is_dates":         is_m["equity_dates"],
                "oos_dates":        oos_m["equity_dates"],
                "optimal_params":   {k: window_cfg.get(k) for k in numeric_keys} if numeric_keys else None,
                # Líquido (fees + funding reais da exchange); None quando custos off.
                "is_net_return":    is_m.get("net_return_pct"),
                "oos_net_return":   oos_m.get("net_return_pct"),
                "oos_net_sharpe":   oos_m.get("net_sharpe"),
                "oos_fees":         oos_m.get("fees_total"),
                "oos_funding":      oos_m.get("funding_total"),
            })

        if not windows:
            return jsonify({"error": "Nenhuma janela gerou trades suficientes. Tente reduzir o numero de janelas."}), 400

        # Concatena a curva OOS reescalando cada segmento para continuidade
        oos_dates_all  = []
        oos_values_all = []
        running_base   = None

        for w in windows:
            raw_vals  = w["oos_equity"]
            raw_dates = w["oos_dates"]
            if not raw_vals:
                continue
            initial = raw_vals[0] if raw_vals[0] else 1.0
            if running_base is None:
                oos_dates_all.extend(raw_dates)
                oos_values_all.extend(raw_vals)
                running_base = next((v for v in reversed(raw_vals) if v is not None), 1.0)
            else:
                scale = (running_base / initial) if initial != 0 else 1.0
                scaled = [v * scale if v is not None else None for v in raw_vals]
                oos_dates_all.extend(raw_dates)
                oos_values_all.extend(scaled)
                # Usa o ultimo valor nao-nulo para evitar que running_base fique None
                running_base = next((v for v in reversed(scaled) if v is not None), running_base)

        # WFE = media(oos_annualized / is_annualized) para janelas com is_annualized > 0
        # Razoes clampadas em [-2, 5] para evitar que outliers distorcam a media
        # WFE > 0.5 e geralmente aceitavel
        wfe_ratios = []
        for w in windows:
            ia = w["is_annualized"]
            oa = w["oos_annualized"]
            if ia is not None and ia > 0 and oa is not None:
                wfe_ratios.append(min(max(oa / ia, -2.0), 5.0))

        wfe = float(np.mean(wfe_ratios)) if wfe_ratios else 0.0

        ann_oos = [w["oos_annualized"] for w in windows if w["oos_annualized"] is not None]
        ann_is  = [w["is_annualized"]  for w in windows if w["is_annualized"]  is not None]
        avg_oos_annualized = float(np.mean(ann_oos)) if ann_oos else 0.0
        avg_is_annualized  = float(np.mean(ann_is))  if ann_is  else 0.0
        avg_oos_return = float(np.mean([w["oos_return"] for w in windows if w["oos_return"] is not None]))
        avg_is_return  = float(np.mean([w["is_return"]  for w in windows if w["is_return"]  is not None]))

        resp = {
            "windows":              windows,
            "oos_equity_curve":     {"dates": oos_dates_all, "values": oos_values_all},
            "wfe":                  _safe(wfe),
            "avg_oos_annualized":   _safe(avg_oos_annualized),
            "avg_is_annualized":    _safe(avg_is_annualized),
            "avg_oos_return":       _safe(avg_oos_return),
            "avg_is_return":        _safe(avg_is_return),
            "n_valid_windows":      len(windows),
            "param_keys":           numeric_keys,
            "costs_applied":        cost_ctx is not None,
            "cost_warnings":        cost_warnings,
        }

        if cost_ctx is not None:
            net_oos = [w["oos_net_return"] for w in windows if w["oos_net_return"] is not None]
            resp.update({
                "cost_exchange":       cost_ctx["exchange"],
                "cost_scenario":       cost_ctx["scenario"],
                "cost_use_funding":    cost_ctx["use_funding"],
                "avg_oos_net_return":  _safe(float(np.mean(net_oos)) if net_oos else 0.0),
                "total_oos_fees":      _safe(float(sum(w["oos_fees"]    or 0.0 for w in windows))),
                "total_oos_funding":   _safe(float(sum(w["oos_funding"] or 0.0 for w in windows))),
            })

        return jsonify(resp)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/backtest/correlation", methods=["POST"])
def api_backtest_correlation():
    """
    Baixa retornos de 2 anos e retorna matriz de correlação + distribuições.
    Usa cache em memória para evitar re-download.
    """
    try:
        import yfinance as yf

        body = request.get_json(force=True) or {}
        tickers = body.get("tickers", {})  # { "Bitcoin (BTC)": "BTC-USD", ... }

        if len(tickers) < 2:
            return jsonify({"error": "Selecione pelo menos 2 ativos"}), 400

        cache_key = frozenset(tickers.items())
        if cache_key in _corr_cache:
            return jsonify(_corr_cache[cache_key])

        # Download dados (lógica de backtest_live.py linhas 348-370)
        closes = {}
        for label, ticker in tickers.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2y", interval="1d")
                if not hist.empty:
                    series = hist["Close"].copy()
                    series.index = series.index.tz_localize(None).normalize()  # type: ignore[union-attr]
                    series = series[~series.index.duplicated(keep="first")]
                    closes[label] = series
            except Exception:
                continue

        if len(closes) < 2:
            return jsonify({"error": "Não foi possível carregar dados para correlação"}), 400

        df_closes = pd.DataFrame(closes).ffill().dropna()
        returns_df = df_closes.pct_change().dropna()

        # Matriz de correlação
        corr = returns_df.corr()
        labels = corr.columns.tolist()
        matrix = [[_safe(float(v)) for v in row] for row in corr.values]

        # Distribuições
        distributions = {}
        colors = [
            "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
            "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
        ]
        for i, col in enumerate(returns_df.columns):
            ret = returns_df[col].dropna()
            skew_val = float(ret.skew())  # type: ignore[arg-type]
            kurt_val = float(ret.kurtosis())  # type: ignore[arg-type]
            sharpe_val = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

            # Histograma
            ret_pct = (ret * 100).tolist()

            # PDF teórico
            mu = float(ret.mean() * 100)
            sigma = float(ret.std() * 100)
            x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200).tolist()
            pdf = scipy_stats.norm.pdf(x_range, mu, sigma).tolist()

            distributions[col] = {
                "color": colors[i % len(colors)],
                "returns": [_safe(float(v)) for v in ret_pct],
                "pdf_x": [_safe(float(v)) for v in x_range],
                "pdf_y": [_safe(float(v)) for v in pdf],
                "stats": {
                    "mean": _safe(float(ret.mean() * 100)),
                    "std": _safe(float(ret.std() * 100)),
                    "skew": _safe(skew_val),
                    "kurtosis": _safe(kurt_val),
                    "sharpe_annual": _safe(sharpe_val),
                },
                "skew_desc": _skew_desc(skew_val),
                "kurt_desc": _kurt_desc(kurt_val),
            }

        # Retornos alinhados para scatter plot par a par (+ datas para hover)
        dates = [d.strftime("%d/%m/%Y") for d in returns_df.index]
        returns_aligned = {}
        for col in returns_df.columns:
            returns_aligned[col] = [_safe(float(v)) for v in (returns_df[col] * 100).tolist()]

        response = {
            "correlation": {"labels": labels, "matrix": matrix},
            "distributions": distributions,
            "returns_aligned": returns_aligned,
            "dates": dates,
        }
        _corr_cache[cache_key] = response
        return jsonify(response)

    except ImportError:
        return jsonify({"error": "yfinance não instalado: pip install yfinance"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/montecarlo", methods=["POST"])
def api_backtest_montecarlo():
    """
    Simulação Monte Carlo (GBM) sobre a equity curve do backtest.
    Recebe: { equity_curve: {dates, values}, num_sims, horizon, seed }
    Retorna: hist_dates, hist_values, proj_dates, p10, p50, p90, reg_line,
             sample_paths, stats
    """
    try:
        body = request.get_json(force=True) or {}
        equity   = body.get("equity_curve", {})
        num_sims = max(1, int(body.get("num_sims", 300)))
        horizon  = int(body.get("horizon", 0))   # 0 = auto
        seed     = int(body.get("seed", 42))

        dates  = equity.get("dates", [])
        values = equity.get("values", [])

        if not values or len(values) < 5:
            return jsonify({"error": "Dados insuficientes para simulação"}), 400

        arr = np.array(values, dtype=float)

        # ── Parâmetros estatísticos ──────────────────────────────────────────
        log_rets = np.diff(np.log(arr))
        log_rets = log_rets[np.isfinite(log_rets)]
        if len(log_rets) < 5:
            return jsonify({"error": "Retornos insuficientes"}), 400

        mu       = float(np.mean(log_rets))
        variance = float(np.var(log_rets))
        sigma    = float(np.std(log_rets))
        S0       = float(arr[-1])

        steps = max(int(len(arr) * 0.75), 30) if horizon == 0 else horizon

        # ── GBM vetorizado com numpy ─────────────────────────────────────────
        rng        = np.random.default_rng(seed)
        Z          = rng.standard_normal((num_sims, steps))
        increments = np.exp((mu - 0.5 * variance) + sigma * Z)
        paths      = np.cumprod(increments, axis=1) * S0          # (N, steps)
        paths      = np.hstack([np.full((num_sims, 1), S0), paths])  # prepend S0 → (N, steps+1)

        # ── Percentis ────────────────────────────────────────────────────────
        p10 = np.percentile(paths, 10, axis=0).tolist()
        p50 = np.percentile(paths, 50, axis=0).tolist()
        p90 = np.percentile(paths, 90, axis=0).tolist()

        # ── Regressão linear sobre a mediana (numpy.polyfit) ─────────────────
        x_idx    = np.arange(steps + 1, dtype=float)
        coeffs   = np.polyfit(x_idx, p50, 1)
        reg_line = (coeffs[0] * x_idx + coeffs[1]).tolist()

        # ── Amostra de caminhos para visualização (30 paths) ─────────────────
        sample_n    = min(30, num_sims)
        sample_idx  = np.linspace(0, num_sims - 1, sample_n, dtype=int)
        sample_paths = [
            [_safe(float(v)) for v in paths[i]]
            for i in sample_idx
        ]

        # ── Datas futuras via pandas (dias úteis) ────────────────────────────
        last_str = dates[-1] if dates else ""
        try:
            last_ts = pd.Timestamp(last_str)
        except Exception:
            day, mo, yr = last_str.split("/")
            last_ts = pd.Timestamp(f"{yr}-{mo}-{day}")

        future_bdays = pd.bdate_range(start=last_ts, periods=steps + 1)[1:]
        proj_dates   = [last_ts.strftime("%Y-%m-%d")] + [
            d.strftime("%Y-%m-%d") for d in future_bdays
        ]

        # ── Janela histórica visível (≤ metade do horizonte) ─────────────────
        hist_window = min(len(arr), math.ceil(steps / 2))
        hist_dates  = dates[-hist_window:]
        hist_values = [_safe(float(v)) for v in arr[-hist_window:]]

        # ── Estatísticas finais ───────────────────────────────────────────────
        finals = paths[:, -1]
        stats  = {
            "median_final": _safe(float(np.percentile(finals, 50))),
            "p10_final":    _safe(float(np.percentile(finals, 10))),
            "p90_final":    _safe(float(np.percentile(finals, 90))),
            "prob_profit":  _safe(float(np.mean(finals > S0) * 100)),
            "mu_daily":     _safe(mu),
            "sigma_daily":  _safe(sigma),
        }

        return jsonify({
            "hist_dates":   hist_dates,
            "hist_values":  hist_values,
            "proj_dates":   proj_dates,
            "p10":          [_safe(v) for v in p10],
            "p50":          [_safe(v) for v in p50],
            "p90":          [_safe(v) for v in p90],
            "reg_line":     [_safe(v) for v in reg_line],
            "sample_paths": sample_paths,
            "stats":        stats,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/backtest/validate", methods=["POST"])
def api_backtest_validate():
    """
    Validação estatística da estratégia via Monte Carlo + Permutation Test.

    Recebe:
      trades         — lista de trades com pnl_pct
      equity_curve   — {dates, values}
      metrics        — métricas do backtest (initial_capital, final_equity, etc.)
      n_sims         — nº de simulações MC (default 500)
      n_perms        — nº de permutações (default 300)
      seed           — semente aleatória

    Retorna:
      original, reshuffle, resample, randomized, return_alteration,
      permutation_test, report
    """
    import sys as _sys
    import os as _os
    _ROOT = _os.path.dirname(_os.path.abspath(__file__))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)

    try:
        from monte_carlo.monte_carlo      import MonteCarlo
        from monte_carlo.permutation_test import PermutationTestEquity
        from monte_carlo.report           import generate as gen_report
    except ImportError as e:
        return jsonify({"error": f"Módulo monte_carlo_project não encontrado: {e}"}), 500

    try:
        body         = request.get_json(force=True) or {}
        trades       = body.get("trades", [])
        equity       = body.get("equity_curve", {})
        metrics_in   = body.get("metrics", {})
        n_sims       = max(50, int(body.get("n_sims", 500)))
        n_perms      = max(50, int(body.get("n_perms", 300)))
        seed         = int(body.get("seed", 42))

        eq_values    = equity.get("values", [])
        eq_dates     = equity.get("dates", [])

        if len(trades) < 5:
            return jsonify({"error": "São necessários pelo menos 5 trades para a validação"}), 400
        if len(eq_values) < 10:
            return jsonify({"error": "Equity curve insuficiente"}), 400

        ic           = float(metrics_in.get("initial_capital", eq_values[0]))
        interval_mc  = body.get("interval", "1d")
        ann_factor   = _BARS_PER_YEAR.get(interval_mc, 252)
        # Prefere inferir barras/ano do span real da equity (24/7 p/ cripto);
        # a tabela fixa assume pregao de bolsa (6,5h x 252d) e subestima o
        # Sharpe de cripto intraday em ~2,3x. Fallback: tabela.
        try:
            span_s = (pd.Timestamp(eq_dates[-1]) - pd.Timestamp(eq_dates[0])).total_seconds()
            years = span_s / (365.25 * 24 * 3600)
            if years > 0 and len(eq_values) > 1:
                ann_factor = len(eq_values) / years
        except Exception:
            pass

        # ── Monte Carlo ──────────────────────────────────────────────────────
        mc = MonteCarlo(initial_capital=ic, seed=seed, ann_factor=ann_factor)

        reshuffle_r         = mc.reshuffle(trades,        n_sims=n_sims)
        resample_r          = mc.resample(trades,         n_sims=n_sims)
        randomized_r        = mc.randomized(trades,       n_sims=n_sims)
        return_alteration_r = mc.return_alteration(eq_values, eq_dates, n_sims=n_sims)

        # ── Permutation Test (equity-curve based) ────────────────────────────
        pt          = PermutationTestEquity(seed=seed, ann_factor=ann_factor)
        perm_result = pt.run(eq_values, trades, n_perms=n_perms)

        # ── Métricas originais enriquecidas ──────────────────────────────────
        arr_eq  = np.array(eq_values, dtype=float)
        rets_eq = np.diff(arr_eq) / np.where(arr_eq[:-1] != 0, arr_eq[:-1], 1.0)
        rets_eq = rets_eq[np.isfinite(rets_eq)]
        _std_eq = float(rets_eq.std(ddof=1)) if len(rets_eq) > 1 else 0.0
        sharpe  = float(rets_eq.mean() / _std_eq * np.sqrt(ann_factor)) if _std_eq > 0 else 0.0

        pnls      = [t.get("pnl_pct", 0) for t in trades]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p <= 0]
        win_rate  = float(metrics_in.get("win_rate", len(wins) / len(pnls) * 100 if pnls else 0))
        avg_win   = float(np.mean(wins))   if wins   else 0.0
        avg_loss  = float(np.mean(losses)) if losses else 0.0
        expectancy = win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss

        # Sortino — annualized by trades_per_year
        _ds_sq  = [min(p, 0) ** 2 for p in pnls]
        _ds_dev = float(np.sqrt(np.mean(_ds_sq))) if _ds_sq else 0.0
        from datetime import datetime as _dt2
        _sort_days = 1
        if len(eq_dates) > 1:
            try:
                _sort_days = max((_dt2.fromisoformat(str(eq_dates[-1])[:10]) - _dt2.fromisoformat(str(eq_dates[0])[:10])).days, 1)
            except Exception:
                _sort_days = max(len(arr_eq) - 1, 1)
        _trades_per_year = len(pnls) / (_sort_days / 365.25) if _sort_days > 0 else len(pnls)
        sortino = float(np.mean(pnls) / _ds_dev * np.sqrt(_trades_per_year)) if _ds_dev > 0 else 0.0

        # Calmar — CAGR uses calendar days derived from equity curve dates
        ic_val = float(metrics_in.get("initial_capital", arr_eq[0] if len(arr_eq) else 1.0))
        if len(eq_dates) > 1:
            try:
                _t0 = _dt2.fromisoformat(str(eq_dates[0])[:10])
                _t1 = _dt2.fromisoformat(str(eq_dates[-1])[:10])
                _cal_days = max((_t1 - _t0).days, 1)
            except Exception:
                _cal_days = max(len(arr_eq) - 1, 1)
        else:
            _cal_days = max(len(arr_eq) - 1, 1)
        cagr = ((arr_eq[-1] / ic_val) ** (365.25 / _cal_days) - 1) * 100 if ic_val > 0 else 0.0
        max_dd_val = float(metrics_in.get("max_dd", 0)) or 0
        calmar = float(cagr / abs(max_dd_val)) if abs(max_dd_val) > 0 else 0.0

        # Omega
        gains_sum  = sum(p for p in pnls if p > 0)
        losses_sum = abs(sum(p for p in pnls if p < 0))
        omega = float(gains_sum / losses_sum) if losses_sum > 0 else 0.0

        # Extract dd episode troughs for Sterling and Burke
        peak_eq = np.maximum.accumulate(arr_eq)
        dd_eq   = (arr_eq - peak_eq) / peak_eq * 100
        _ep_tr  = []
        _in_dd  = False
        _ep_s   = None
        for _ki, _vi in enumerate(dd_eq):
            if _vi < 0 and not _in_dd:
                _in_dd = True
                _ep_s  = _ki
            elif _vi >= 0 and _in_dd:
                _in_dd = False
                _ep_tr.append(float(dd_eq[_ep_s:_ki].min()))
        if _in_dd and _ep_s is not None:
            _ep_tr.append(float(dd_eq[_ep_s:].min()))

        # Sterling (CAGR / mean of N worst episode troughs)
        if _ep_tr:
            _nw = min(5, len(_ep_tr))
            _wt = sorted(_ep_tr)[:_nw]
            _aw = abs(float(np.mean(_wt)))
            sterling = float(cagr / _aw) if _aw > 0 else 0.0
        else:
            sterling = 0.0

        # Burke (CAGR / sqrt(sum of N worst episode trough^2))
        if _ep_tr:
            _nw = min(5, len(_ep_tr))
            _wt = sorted(_ep_tr)[:_nw]
            _bd = float(np.sqrt(np.sum(np.array(_wt) ** 2)))
            burke = float(cagr / _bd) if _bd > 0 else 0.0
        else:
            burke = 0.0

        original = {
            **{k: _safe(v) for k, v in metrics_in.items()},
            "sharpe":     _safe(sharpe),
            "sortino":    _safe(sortino),
            "calmar":     _safe(calmar),
            "omega":      _safe(omega),
            "sterling":   _safe(sterling),
            "burke":      _safe(burke),
            "avg_win":    _safe(avg_win),
            "avg_loss":   _safe(avg_loss),
            "expectancy": _safe(expectancy),
        }

        # ── Relatório textual ────────────────────────────────────────────────
        report_text = gen_report(
            original=original,
            mc_results={
                "reshuffle":        reshuffle_r,
                "resample":         resample_r,
                "randomized":       randomized_r,
                "return_alteration": return_alteration_r,
            },
            perm_result=perm_result,
            strategy_label=body.get("strategy_label", "Estratégia"),
            asset_label=body.get("asset_label", "-"),
            period_label=f"{eq_dates[0] if eq_dates else '-'} → {eq_dates[-1] if eq_dates else '-'}",
            n_sims=n_sims,
        )

        return jsonify({
            "original":          original,
            "reshuffle":         reshuffle_r,
            "resample":          resample_r,
            "randomized":        randomized_r,
            "return_alteration": return_alteration_r,
            "permutation_test":  perm_result,
            "report":            report_text,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ─── Optimizer ────────────────────────────────────────────────────────────────

# Estado global do otimizador (cancel + progresso)
_optimizer_cancel = threading.Event()
_optimizer_progress = {"current": 0, "total": 0, "valid": 0, "status": "idle"}
_optimizer_lock = threading.Lock()
_optimizer_result_store = {"data": None, "error": None}

# Limites do brute force. O contador nunca deve expandir o produto cartesiano
# apenas para descobrir que o grid era grande demais.
_OPTIMIZER_MAX_COMBINATIONS = 2_000_000
_OPTIMIZER_EXACT_COUNT_LIMIT = 100_000


def _parse_grid_generic(grid_raw, schema):
    """Converte valores do grid usando o CONFIG_SCHEMA da estrategia para tipos corretos."""
    type_map = {}
    for section in (schema or []):
        for field in section.get("fields", []):
            type_map[field["key"]] = field.get("type", "number")

    typed = {}
    for key, values in grid_raw.items():
        if not isinstance(values, list):
            continue
        if len(values) == 0:
            # Uma selecao explicitamente vazia torna o grid invalido. Antes a
            # chave era omitida e podia herdar silenciosamente um valor antigo.
            typed[key] = []
            continue
        ft = type_map.get(key)
        if ft is None:
            # Infere tipo a partir dos valores
            ft = _infer_grid_type(values)
        if ft == "checkbox":
            typed[key] = [v if isinstance(v, bool) else str(v).lower() in ("true", "1") for v in values]
        elif ft == "number":
            typed[key] = [float(v) for v in values]
        elif ft == "select":
            typed[key] = [str(v) for v in values]
        else:
            typed[key] = values
    return typed


def _infer_grid_type(values):
    """Infere o tipo de um campo a partir dos valores no grid."""
    if all(isinstance(v, bool) or str(v).lower() in ("true", "false") for v in values):
        return "checkbox"
    try:
        [float(v) for v in values]
        return "number"
    except (ValueError, TypeError):
        return "select"


def _grid_product_size(grid):
    """Tamanho bruto do produto cartesiano, sem alocar as combinacoes."""
    if not grid:
        return 1
    total = 1
    for values in grid.values():
        if not values:
            return 0
        total *= len(values)
    return total


def _iter_combos(grid):
    """Itera o produto cartesiano sem materializa-lo em memoria."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _optimizer_combo_count(module, grid, fixed_params=None):
    """Retorna (count, exact, raw_count) sem expandir grids gigantes."""
    raw_count = _grid_product_size(grid)
    if raw_count == 0:
        return 0, True, 0

    strategy_counter = getattr(module, "count_optimizer_configs", None)
    if strategy_counter:
        return int(strategy_counter(grid, fixed_params or {})), True, raw_count

    validator = getattr(module, "is_valid_config", None)
    if validator and raw_count <= _OPTIMIZER_EXACT_COUNT_LIMIT:
        count = sum(1 for params in _iter_combos(grid) if validator(params))
        return count, True, raw_count

    # Para estrategias sem contador condicional, o bruto e um limite superior.
    # Ainda e muito mais util que travar a API tentando obter o valor exato.
    return raw_count, validator is None, raw_count


def _iter_optimizer_combos(module, grid, fixed_params=None):
    strategy_iterator = getattr(module, "iter_optimizer_configs", None)
    if strategy_iterator:
        yield from strategy_iterator(grid, fixed_params or {})
    else:
        yield from _iter_combos(grid)


def _calc_optimizer_row(result, params, param_labels):
    """Extrai metricas padrao do resultado de module.run() e monta linha para ranking."""
    m = result.get("metrics", {})
    trades = result.get("trades", [])

    total_return = m.get("total_return", 0) or 0
    max_dd = m.get("max_dd", 0) or 0
    win_rate = m.get("win_rate", 0) or 0
    pf = m.get("profit_factor", 0) or 0
    avg_win = m.get("avg_win", 0) or 0
    avg_loss = m.get("avg_loss", 0) or 0

    # Reuse pre-computed formula-accurate values from the strategy module
    sharpe  = m.get("sharpe")  or 0.0
    sortino = m.get("sortino") or 0.0
    calmar  = m.get("calmar")  or 0.0
    omega   = m.get("omega")   or 0.0
    sterling = m.get("sterling") or 0.0
    burke   = m.get("burke")   or 0.0

    # Score composto
    score = total_return * (win_rate / 100) / max(abs(max_dd), 1)

    row = {
        "Retorno (%)": round(total_return, 2),
        "Max DD (%)": round(max_dd, 2),
        "Trades": m.get("total_trades", 0),
        "Win Rate (%)": round(win_rate, 1),
        "Avg Win (%)": round(avg_win, 2),
        "Avg Loss (%)": round(avg_loss, 2),
        "Profit Factor": round(pf, 2),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
        "Omega": round(omega, 2),
        "Sterling": round(sterling, 2),
        "Burke": round(burke, 2),
        "Score": round(score, 2),
    }
    # Adiciona parametros com labels amigaveis
    for k, v in params.items():
        label = param_labels.get(k, k)
        row[label] = v
    return row


def _build_param_labels(schema):
    """Monta mapa key -> label para colunas do CSV do otimizador.

    Usa os nomes de coluna do COLUMN_MAP (que o dashboard espera) quando
    disponivel, senao usa o label do CONFIG_SCHEMA.
    """
    # Mapa inverso: optimizer_key -> csv_column_key (ex: ma_type -> ma)
    opt_to_csv = {v: k for k, v in DASHBOARD_TO_OPTIMIZER_KEY.items()}
    labels = {}
    for section in (schema or []):
        for field in section.get("fields", []):
            key = field["key"]
            csv_key = opt_to_csv.get(key)
            if csv_key and csv_key in COLUMN_DISPLAY:
                labels[key] = COLUMN_DISPLAY[csv_key]
            else:
                labels[key] = field.get("label", key)
    return labels


@app.route("/api/optimizer/grids", methods=["GET"])
def api_optimizer_grids():
    """Retorna os grids disponiveis para uma estrategia."""
    strategy_file = request.args.get("strategy", "depaula")
    try:
        module = _load_strategy(strategy_file)
        raw_grids = getattr(module, "OPTIMIZER_GRIDS", {})
        grids_info = {}
        for name, grid in raw_grids.items():
            grids_info[name] = {k: [str(v) for v in vals] for k, vals in grid.items()}
        schema = getattr(module, "CONFIG_SCHEMA", [])
        return jsonify({"grids": grids_info, "schema": schema})
    except Exception as e:
        return jsonify({"grids": {}, "schema": [], "error": str(e)})


@app.route("/api/optimizer/count", methods=["POST"])
def api_optimizer_count():
    """Conta combinacoes sem materializar o produto cartesiano inteiro."""
    try:
        body = request.get_json(force=True) or {}
        grid_raw = body.get("grid", {})
        strategy_file = body.get("strategy_file", "depaula")

        module = _load_strategy(strategy_file)
        schema = getattr(module, "CONFIG_SCHEMA", [])
        typed_grid = _parse_grid_generic(grid_raw, schema)
        fixed_params = dict(body.get("config") or {})
        count, exact, raw_count = _optimizer_combo_count(
            module, typed_grid, fixed_params)
        too_many = count > _OPTIMIZER_MAX_COMBINATIONS
        message = None
        if count == 0:
            message = ("Nenhuma combinacao valida. Selecione ao menos um valor "
                       "para cada parametro.")
        elif too_many:
            message = (
                f"Grid com {count:,} combinacoes; o limite e "
                f"{_OPTIMIZER_MAX_COMBINATIONS:,}. Reduza os intervalos."
            )
        return jsonify({
            "count": count,
            "exact": exact,
            "raw_count": raw_count,
            "too_many": too_many,
            "max_combinations": _OPTIMIZER_MAX_COMBINATIONS,
            "message": message,
        })
    except Exception as e:
        return jsonify({"error": str(e), "count": 0}), 400


@app.route("/api/optimizer/stop", methods=["POST"])
def api_optimizer_stop():
    """Para a otimizacao em andamento e retorna resultados parciais."""
    _optimizer_cancel.set()
    return jsonify({"ok": True})


@app.route("/api/optimizer/progress", methods=["GET"])
def api_optimizer_progress():
    """Retorna o progresso atual da otimizacao."""
    with _optimizer_lock:
        return jsonify(dict(_optimizer_progress))


def _run_optimizer_compute(df_data, module, grid_raw, capital, min_trades, rank_by, top_n, symbol_label, interval_label, fixed_params=None):
    """Logica generica de otimizacao. Retorna (data_dict, error_str) — sem jsonify."""
    import time

    schema = getattr(module, "CONFIG_SCHEMA", [])
    validator = getattr(module, "is_valid_config", None)
    prepare = getattr(module, "prepare_optimizer_params", None)
    param_labels = _build_param_labels(schema)

    typed_grid = _parse_grid_generic(grid_raw, schema)
    semantic_count, _exact, raw_count = _optimizer_combo_count(
        module, typed_grid, fixed_params)
    has_strategy_iterator = callable(getattr(module, "iter_optimizer_configs", None))
    # Sem iterador condicional, o loop ainda visita o produto bruto e o progresso
    # deve refletir isso, mesmo quando o endpoint de contagem calculou os validos.
    total = semantic_count if has_strategy_iterator else raw_count

    if semantic_count == 0:
        return None, "Nenhuma combinacao gerada. Verifique o grid."
    if total > _OPTIMIZER_MAX_COMBINATIONS:
        return None, (
            f"Grid muito grande ({total:,} combinacoes; bruto {raw_count:,}). "
            f"Limite: {_OPTIMIZER_MAX_COMBINATIONS:,}. Reduza os parametros."
        )

    # Limpa cancel ANTES de marcar running — evita race com thread anterior
    _optimizer_cancel.clear()
    with _optimizer_lock:
        _optimizer_progress["current"] = 0
        _optimizer_progress["total"] = total
        _optimizer_progress["valid"] = 0
        _optimizer_progress["status"] = "running"

    base_extra = dict(fixed_params) if fixed_params else {"initial_capital": capital}
    results = []
    t0 = time.time()
    stopped = False

    for i, params in enumerate(
            _iter_optimizer_combos(module, typed_grid, fixed_params), start=1):
        if _optimizer_cancel.is_set():
            stopped = True
            break
        if validator and not validator(params):
            with _optimizer_lock:
                _optimizer_progress["current"] = i
            continue
        try:
            run_params = {**base_extra, **params, "_fast": True}
            if prepare:
                run_params = prepare(dict(run_params))

            result = module.run(df_data, run_params)
            # Exibe apenas as chaves escolhidas no grid. Os hooks de estrategia
            # podem acrescentar defaults canonicos para executar corretamente.
            display_params = {key: params.get(key) for key in typed_grid}
            row = _calc_optimizer_row(result, display_params, param_labels)
            if row and row["Trades"] >= min_trades:
                results.append(row)
        except Exception:
            pass
        with _optimizer_lock:
            _optimizer_progress["current"] = i
            _optimizer_progress["valid"] = len(results)

    elapsed = time.time() - t0
    tested = _optimizer_progress["current"]

    if not results:
        return {
            "results": [], "total_tested": tested, "valid_count": 0,
            "elapsed": round(elapsed, 1), "best": None, "stopped": stopped,
        }, None

    results_df = pd.DataFrame(results)
    from datetime import datetime as _dt
    results_df.insert(0, "Data", _dt.now().strftime("%Y-%m-%d %H:%M"))
    results_df.insert(1, "Ativo", symbol_label)
    results_df.insert(2, "Timeframe", interval_label)
    results_df = results_df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "Rank"

    DATA_DIR.mkdir(exist_ok=True)
    safe_symbol = symbol_label.replace(" ", "_").replace("/", "_")
    csv_filename = f"otimizacao_{safe_symbol}_{interval_label}.csv"
    csv_path = DATA_DIR / csv_filename
    results_df.to_csv(csv_path, index=True, index_label="Rank", encoding="utf-8-sig")

    top_results = results_df.head(top_n)
    all_records = []
    for idx, row_data in top_results.iterrows():
        record = {k: _safe(v) if isinstance(v, float) else v for k, v in row_data.to_dict().items()}
        record["Rank"] = idx
        all_records.append(record)

    best_row = results_df.iloc[0]
    best = {k: _safe(v) if isinstance(v, float) else v for k, v in best_row.to_dict().items()}
    csv_string = results_df.to_csv(index=True, index_label="Rank", encoding="utf-8-sig")

    metric_cols = ["Retorno (%)", "Max DD (%)", "Trades", "Win Rate (%)", "Profit Factor", "Sharpe", "Score"]
    param_cols = [param_labels.get(k, k) for k in sorted(typed_grid.keys())]

    return {
        "results": all_records,
        "total_tested": tested,
        "valid_count": len(results),
        "elapsed": round(elapsed, 1),
        "best": best,
        "csv_filename": csv_filename,
        "csv_data": csv_string,
        "symbol": symbol_label,
        "interval": interval_label,
        "metric_columns": metric_cols,
        "param_columns": param_cols,
        "stopped": stopped,
    }, None


def _optimizer_worker(df_data, module, grid_raw, capital, min_trades, rank_by, top_n, symbol_label, interval_label, fixed_params):
    """Executa a otimizacao em background thread e armazena o resultado."""
    try:
        data, err = _run_optimizer_compute(
            df_data, module, grid_raw, capital, min_trades, rank_by, top_n,
            symbol_label, interval_label, fixed_params,
        )
        if err:
            _optimizer_result_store["data"] = None
            _optimizer_result_store["error"] = err
            with _optimizer_lock:
                _optimizer_progress["status"] = "error"
        else:
            _optimizer_result_store["data"] = data
            _optimizer_result_store["error"] = None
            with _optimizer_lock:
                _optimizer_progress["status"] = "done"
    except Exception as e:
        import traceback as _tb
        _optimizer_result_store["data"] = None
        _optimizer_result_store["error"] = str(e) + "\n" + _tb.format_exc()
        with _optimizer_lock:
            _optimizer_progress["status"] = "error"


@app.route("/api/optimizer/result", methods=["GET"])
def api_optimizer_result():
    """Retorna o resultado armazenado da ultima otimizacao."""
    with _optimizer_lock:
        status = _optimizer_progress.get("status", "idle")
    if status in ("starting", "running"):
        return jsonify({"status": status}), 202
    if _optimizer_result_store["error"]:
        return jsonify({"error": _optimizer_result_store["error"]}), 500
    if _optimizer_result_store["data"] is None:
        return jsonify({"error": "Nenhum resultado disponivel"}), 404
    return jsonify(_optimizer_result_store["data"])


@app.route("/api/optimizer/run", methods=["POST"])
def api_optimizer_run():
    """Inicia a otimizacao em background e retorna imediatamente."""
    try:
        body = request.get_json(force=True) or {}
        strategy_file = body.get("strategy_file", "depaula")
        symbol = body.get("symbol", "BTC-USD")
        symbol_label = body.get("symbol_label", symbol)
        interval = body.get("interval", "1d")
        grid_raw = body.get("grid", {})
        capital = float(body.get("capital", 1000.0))
        min_trades = int(body.get("min_trades", 5))
        rank_by = body.get("rank_by", "Score")
        top_n = int(body.get("top_n", 20))

        if body.get("data_source", "asset") != "asset":
            return jsonify({"error": "Use upload CSV via multipart"}), 400

        # Merge backtest config so non-grid params match the user's backtest settings.
        # Grid combos override grid params; everything else comes from config.
        fixed_params = dict(body.get("config", {}))
        fixed_params["initial_capital"] = capital
        if body.get("cycle_long_months"):
            fixed_params["cycle_long_months"] = body["cycle_long_months"]
        if body.get("cycle_short_months"):
            fixed_params["cycle_short_months"] = body["cycle_short_months"]

        df_data = _download_data_safe(symbol, interval, body.get("exchange"))

        # Filtra por data se o usuario definiu start/end
        sd = body.get("start_date")
        ed = body.get("end_date")
        if sd:
            df_data = df_data[df_data.index >= pd.Timestamp(sd)]
        if ed:
            df_data = df_data[df_data.index <= pd.Timestamp(ed)]
        if df_data.empty:
            return jsonify({"error": "Nenhum dado no intervalo de datas selecionado"}), 400

        module = _load_strategy(strategy_file)

        _optimizer_result_store["data"] = None
        _optimizer_result_store["error"] = None
        with _optimizer_lock:
            _optimizer_progress["status"] = "starting"

        t = threading.Thread(
            target=_optimizer_worker,
            args=(df_data, module, grid_raw, capital, min_trades, rank_by, top_n,
                  symbol_label, interval, fixed_params),
            daemon=True,
        )
        t.start()

        return jsonify({"status": "started"})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/optimizer/run-csv", methods=["POST"])
def api_optimizer_run_csv():
    """Inicia otimizacao com CSV em background e retorna imediatamente."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Arquivo CSV obrigatorio"}), 400

        file_obj = request.files["file"]
        strategy_file = request.form.get("strategy_file", "depaula")
        grid_raw = json.loads(request.form.get("grid", "{}"))
        capital = float(request.form.get("capital", 1000.0))
        min_trades = int(request.form.get("min_trades", 5))
        rank_by = request.form.get("rank_by", "Score")
        top_n = int(request.form.get("top_n", 20))

        raw = file_obj.read()
        df_data = None
        for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                text = raw.decode(enc)
                df_data = pd.read_csv(
                    io.StringIO(text), parse_dates=["Date"], index_col="Date"
                ).sort_index()
                break
            except Exception:
                continue

        if df_data is None:
            return jsonify({"error": "Nao foi possivel ler o CSV"}), 400

        # Merge backtest config so non-grid params match the user's settings
        config_raw = request.form.get("config")
        fixed_params = json.loads(config_raw) if config_raw else {}
        fixed_params["initial_capital"] = capital
        cycle_long = request.form.get("cycle_long_months")
        cycle_short = request.form.get("cycle_short_months")
        if cycle_long:
            fixed_params["cycle_long_months"] = json.loads(cycle_long)
        if cycle_short:
            fixed_params["cycle_short_months"] = json.loads(cycle_short)

        module = _load_strategy(strategy_file)

        _optimizer_result_store["data"] = None
        _optimizer_result_store["error"] = None
        with _optimizer_lock:
            _optimizer_progress["status"] = "starting"

        t = threading.Thread(
            target=_optimizer_worker,
            args=(df_data, module, grid_raw, capital, min_trades, rank_by, top_n,
                  file_obj.filename or "CSV", "-", fixed_params),
            daemon=True,
        )
        t.start()

        return jsonify({"status": "started"})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500




def _skew_desc(val: float) -> str:
    if val < -0.5:
        return "cauda pesada para perdas"
    elif val < -0.1:
        return "leve vies negativo"
    elif val > 0.5:
        return "cauda pesada para ganhos"
    elif val > 0.1:
        return "leve vies positivo"
    return "neutra"


def _kurt_desc(val: float) -> str:
    if val > 3:
        return "caudas muito pesadas"
    elif val > 1:
        return "caudas pesadas"
    elif val < -0.5:
        return "caudas leves"
    return "normal"


# Prop Challenge
@app.route("/api/prop-challenge/simulate", methods=["POST"])
def api_prop_challenge_simulate():
    """
    Simula desafio de mesa prop usando trades de um backtest.
    Faz Monte Carlo reamostrando trades para estimar probabilidade de aprovacao.

    Body JSON:
    {
      "strategy_file": "depaula",
      "data_source": "asset",
      "symbol": "BTC-USD",
      "symbol_label": "Bitcoin",
      "interval": "1d",
      "config": { ... },
      "account_size": 50000,
      "num_sims": 1000
    }
    """
    try:
        body = request.get_json(force=True) or {}
        strategy_file = body.get("strategy_file", "depaula") or "depaula"
        cfg_dict = body.get("config", {})
        account_size = float(body.get("account_size", 50000))
        num_sims = max(100, min(int(body.get("num_sims", 1000)), 10000))
        symbol = body.get("symbol", "")
        symbol_label = body.get("symbol_label", symbol)
        interval_label = body.get("interval", "1d")

        if not symbol:
            return jsonify({"error": "symbol obrigatorio"}), 400
        if account_size <= 0:
            return jsonify({"error": "account_size deve ser positivo"}), 400

        # Forca initial_capital = account_size para o backtest
        cfg_dict["initial_capital"] = account_size

        df_data = _download_data_safe(symbol, interval_label, body.get("exchange"))
        module = _load_strategy(strategy_file)
        result_dict = module.run(df_data.copy(), cfg_dict)

        trades = result_dict.get("trades", [])
        if len(trades) < 5:
            return jsonify({"error": "Poucos trades para simular (minimo 5)"}), 400

        pnl_pcts = [float(t["pnl_pct"]) for t in trades if t.get("pnl_pct") is not None]
        if len(pnl_pcts) < 5:
            return jsonify({"error": "Poucos trades validos para simular (minimo 5)"}), 400

        valid_trades = [t for t in trades if t.get("pnl_pct") is not None]

        # ── Sizing por risco fixo (opcional) ──────────────────────────────
        # Redimensiona cada trade para arriscar `risk_pct` da conta na
        # distância REAL do stop: alavancagem implícita = risco ÷ stop_dist
        # (limitada por lev_cap). Fees maker/taker e funding ESPERADO por
        # período de 8h segurado são descontados por trade já na alavancagem
        # nova. Substitui o módulo de custos históricos (evita dupla contagem).
        rs = body.get("risk_sizing") or {}
        risk_summary = None
        pool = pnl_pcts
        if rs.get("enabled"):
            risk_pct = float(rs.get("risk_pct", 1.0))
            lev_cap = max(float(rs.get("lev_cap", 10.0)), 0.01)
            fee_maker = float(rs.get("fee_maker_pct", 0.02))
            fee_taker = float(rs.get("fee_taker_pct", 0.055))
            funding_8h = float(rs.get("funding_8h_pct", 0.0032))
            maker_entry = bool(rs.get("maker_entry", True))
            resized, levs = [], []
            capped = skipped = 0
            fee_drag_sum = funding_sum = 0.0
            for t in valid_trades:
                entry = t.get("entry_price")
                exitp = t.get("exit_price")
                stop = t.get("stop_price")
                side = int(t.get("direction") or 1)
                stop_dist = (abs(entry - stop) / entry * 100
                             if entry and stop and entry > 0 else None)
                if not entry or not exitp or not stop_dist or stop_dist <= 0:
                    # sem stop definido não dá para dimensionar por risco:
                    # mantém o pnl nativo e sinaliza
                    skipped += 1
                    resized.append(float(t["pnl_pct"]))
                    levs.append(float(t.get("leverage") or 1.0))
                    continue
                exp = risk_pct / stop_dist
                if exp > lev_cap:
                    exp = lev_cap
                    capped += 1
                move = side * (exitp / entry - 1) * 100
                comment = (t.get("exit_comment") or "").lower()
                exit_maker = "maker" in comment or "alvo" in comment
                fees = ((fee_maker if maker_entry else fee_taker)
                        + (fee_maker if exit_maker else fee_taker)) * exp
                hold_h = max(((t.get("exit_ts") or 0) - (t.get("entry_ts") or 0))
                             / 3.6e6, 0.0)
                # long paga funding médio positivo; short recebe
                funding = funding_8h * (hold_h / 8.0) * exp * side
                resized.append(exp * move - fees - funding)
                levs.append(exp)
                fee_drag_sum += fees
                funding_sum += funding
            pool = resized
            n = len(resized) or 1
            risk_summary = {
                "applied": True, "risk_pct": risk_pct, "lev_cap": lev_cap,
                "fee_maker_pct": fee_maker, "fee_taker_pct": fee_taker,
                "funding_8h_pct": funding_8h, "maker_entry": maker_entry,
                "avg_leverage": round(float(np.mean(levs)), 2),
                "max_leverage": round(float(np.max(levs)), 2),
                "pct_capped": round(capped / n * 100, 1),
                "skipped": skipped,
                "avg_fee_drag_pct": round(fee_drag_sum / n, 4),
                "avg_funding_drag_pct": round(funding_sum / n, 4),
                "avg_gross_pnl": round(float(np.mean(pnl_pcts)), 4),
                "avg_net_pnl": round(float(np.mean(resized)), 4),
                "note": ("alavancagem implícita = risco ÷ distância do stop; "
                         "fees e funding esperados já descontados por trade — "
                         "o módulo de custos históricos é ignorado neste modo"),
            }

        # Custos reais da corretora (fees + funding) descontados de cada trade.
        # Quando ligado, o Monte Carlo reamostra o pool LÍQUIDO em vez do bruto.
        # (desativado quando o sizing por risco fixo já embute custos esperados)
        cost_ctx, cost_warnings = ((None, [])
                                   if risk_summary
                                   else _build_wfa_cost_ctx(df_data, body, cfg_dict))
        cost_summary = None
        if cost_ctx is not None:
            net = _net_pnl_pcts(trades, cost_ctx)
            if net["net_pcts"]:
                pool = net["net_pcts"]
            cost_summary = {
                "applied": True,
                "exchange": cost_ctx["exchange"],
                "scenario": cost_ctx["scenario"],
                "use_funding": cost_ctx["use_funding"],
                "use_maker_entry": cost_ctx["use_maker_entry"],
                "use_maker_exit": cost_ctx["use_maker_exit"],
                "total_fees": round(net["total_fees"], 2),
                "total_funding": round(net["total_funding"], 2),
                "avg_gross_pnl": round(float(np.mean(pnl_pcts)), 4),
                "avg_net_pnl": round(float(np.mean(pool)), 4),
                "warnings": cost_warnings,
            }

        rng = np.random.default_rng()  # semente aleatoria — cada execucao produz resultado diferente

        # Regras do desafio
        phase1_target = 0.10   # +10%
        phase2_target = 0.05   # +5%
        max_loss = -0.10       # -10% total
        daily_max_loss = -0.05 # -5% acumulado em um dia

        # Pool de DIAS: agrupa os trades (na ordem de execucao) pelo dia de
        # entrada e reamostra dias inteiros. Preserva a correlacao intradiaria
        # (perdas agrupadas em dias ruins) que a reamostragem iid de trades
        # destruia, e permite acumular a perda diaria de verdade — antes um
        # unico trade <= -5% reprovava, mas varios trades pequenos somando -5%
        # no mesmo dia passavam despercebidos (irreal p/ estrategia intradiaria).
        day_groups = {}
        for pos, (t, p) in enumerate(zip(valid_trades, pool)):
            try:
                key = str(pd.Timestamp(t.get("entry_date", "")).date())
            except Exception:
                key = f"_seq_{pos}"          # sem data parseavel: vira um "dia" proprio
            day_groups.setdefault(key, []).append(float(p))
        day_pool = [day_groups[k] for k in sorted(day_groups)]

        def simulate_phase(days_pool, target, starting_balance):
            """Simula uma fase reamostrando dias inteiros de trading.
            Violacao da perda diaria acumulada (<= -5% no dia) reprova, como
            nas regras reais. Retorna (passed, final_balance, curve, days_used).
            Limitacao conhecida: drawdown intra-trade nao e observado."""
            balance = starting_balance
            curve = [balance]

            max_days = 730  # limite generoso p/ estrategias lentas
            for d in range(1, max_days + 1):
                day = days_pool[int(rng.integers(len(days_pool)))]
                day_start = balance
                for trade_pnl_pct in day:
                    balance += balance * (trade_pnl_pct / 100.0)
                    curve.append(balance)

                    # Perda total acumulada desde o inicio da fase
                    total_change = (balance - starting_balance) / starting_balance
                    if total_change <= max_loss:
                        return False, balance, curve, d

                    # Perda diaria ACUMULADA (soma dos trades do dia)
                    if (balance - day_start) / day_start <= daily_max_loss:
                        return False, balance, curve, d

                    # Alvo da fase
                    if total_change >= target:
                        return True, balance, curve, d

            # Nao atingiu o alvo em max_days
            return False, balance, curve, max_days

        # Monte Carlo
        phase1_pass = 0
        phase2_pass = 0
        both_pass = 0
        phase1_curves = []
        phase2_curves = []
        phase1_results = []
        phase2_results = []

        for i in range(num_sims):
            # Fase 1
            p1_passed, p1_balance, p1_curve, p1_days = simulate_phase(
                day_pool, phase1_target, account_size
            )
            phase1_results.append({
                "passed": p1_passed,
                "final_balance": round(p1_balance, 2),
                "pnl_pct": round((p1_balance - account_size) / account_size * 100, 2),
                "num_trades": len(p1_curve) - 1,
                "days": p1_days,
            })

            if p1_passed:
                phase1_pass += 1
                # Fase 2: comeca com o saldo da conta original (reseta)
                p2_passed, p2_balance, p2_curve, p2_days = simulate_phase(
                    day_pool, phase2_target, account_size
                )
                phase2_results.append({
                    "passed": p2_passed,
                    "final_balance": round(p2_balance, 2),
                    "pnl_pct": round((p2_balance - account_size) / account_size * 100, 2),
                    "num_trades": len(p2_curve) - 1,
                    "days": p2_days,
                })
                if p2_passed:
                    phase2_pass += 1
                    both_pass += 1

            # Salva ate 50 curvas de exemplo para o grafico
            if len(phase1_curves) < 50:
                phase1_curves.append([round(v, 2) for v in p1_curve])
            if p1_passed and len(phase2_curves) < 50:
                phase2_curves.append([round(v, 2) for v in p2_curve])

        # Estatisticas dos trades efetivamente simulados (liquido quando custos on)
        wins = [p for p in pool if p > 0]
        losses = [p for p in pool if p <= 0]
        win_rate = len(wins) / len(pool) * 100 if pool else 0
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0

        # Frequencia media de trades (dias entre trades)
        avg_days_between_trades = None
        trade_dates = []
        for t in trades:
            d = t.get("entry_date", "")
            if d:
                try:
                    trade_dates.append(pd.Timestamp(d))
                except Exception:
                    pass
        if len(trade_dates) >= 2:
            trade_dates.sort()
            total_span = (trade_dates[-1] - trade_dates[0]).days
            avg_days_between_trades = total_span / (len(trade_dates) - 1) if len(trade_dates) > 1 else None

        # Tempo estimado de aprovacao por fase (em dias de calendario).
        # A simulacao conta dias COM trade; converte para calendario pela
        # razao span_total / dias_com_trade do historico.
        p1_passed_trades = [r["num_trades"] for r in phase1_results if r["passed"]]
        p2_passed_trades = [r["num_trades"] for r in phase2_results if r["passed"]]
        p1_passed_days = [r["days"] for r in phase1_results if r["passed"]]
        p2_passed_days = [r["days"] for r in phase2_results if r["passed"]]

        p1_avg_trades = float(np.mean(p1_passed_trades)) if p1_passed_trades else None
        p2_avg_trades = float(np.mean(p2_passed_trades)) if p2_passed_trades else None
        p1_median_trades = float(np.median(p1_passed_trades)) if p1_passed_trades else None
        p2_median_trades = float(np.median(p2_passed_trades)) if p2_passed_trades else None

        cal_factor = 1.0
        if len(trade_dates) >= 2 and len(day_pool) > 0:
            span_days = max((trade_dates[-1] - trade_dates[0]).days, 1) + 1
            cal_factor = max(span_days / len(day_pool), 1.0)

        def _estimate_days(days_used):
            if not days_used:
                return None
            return round(float(np.median(days_used)) * cal_factor, 1)

        p1_est_days = _estimate_days(p1_passed_days)
        p2_est_days = _estimate_days(p2_passed_days)
        total_est_days = None
        if p1_est_days is not None and p2_est_days is not None:
            total_est_days = round(p1_est_days + p2_est_days, 1)

        return jsonify({
            "account_size": account_size,
            "num_sims": num_sims,
            "total_trades": len(trades),
            "symbol": symbol_label,
            "interval": interval_label,
            "strategy": strategy_file,
            "trade_stats": {
                "total": len(pool),
                "win_rate": round(win_rate, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "avg_pnl": round(float(np.mean(pool)), 4),
                "avg_days_between_trades": round(avg_days_between_trades, 1) if avg_days_between_trades else None,
            },
            "costs": cost_summary,
            "risk_sizing": risk_summary,
            "phase1": {
                "target_pct": phase1_target * 100,
                "max_loss_pct": abs(max_loss) * 100,
                "daily_max_loss_pct": abs(daily_max_loss) * 100,
                "passed": phase1_pass,
                "failed": num_sims - phase1_pass,
                "pass_rate": round(phase1_pass / num_sims * 100, 2),
                "avg_trades_to_pass": round(p1_avg_trades, 1) if p1_avg_trades else None,
                "median_trades_to_pass": round(p1_median_trades, 1) if p1_median_trades else None,
                "est_days": p1_est_days,
            },
            "phase2": {
                "target_pct": phase2_target * 100,
                "max_loss_pct": abs(max_loss) * 100,
                "daily_max_loss_pct": abs(daily_max_loss) * 100,
                "passed": phase2_pass,
                "failed": phase1_pass - phase2_pass,
                "pass_rate": round(phase2_pass / phase1_pass * 100, 2) if phase1_pass > 0 else 0,
                "avg_trades_to_pass": round(p2_avg_trades, 1) if p2_avg_trades else None,
                "median_trades_to_pass": round(p2_median_trades, 1) if p2_median_trades else None,
                "est_days": p2_est_days,
            },
            "overall": {
                "passed": both_pass,
                "pass_rate": round(both_pass / num_sims * 100, 2),
                "est_total_days": total_est_days,
            },
            "phase1_curves": phase1_curves,
            "phase2_curves": phase2_curves,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ─── Regime Detection API ────────────────────────────────────────────────────

@app.route("/api/regime/detect", methods=["POST"])
def regime_detect():
    """Detecta regimes de mercado via HMM, Markov Switching ou Change-Point."""
    try:
        from engine.regime_detection import detect_regimes

        content_type = request.content_type or ""

        if "multipart/form-data" in content_type:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Nenhum arquivo enviado"}), 400
            raw = file.read().decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(raw), parse_dates=True, index_col=0)
            params = json.loads(request.form.get("params", "{}"))
        else:
            body = request.get_json(force=True)
            source = body.get("source", "asset")
            params = body.get("params", {})

            if source == "asset":
                symbol = body.get("symbol")
                interval = body.get("interval", "1d")
                if not symbol:
                    return jsonify({"error": "Simbolo nao informado"}), 400
                df = _download_data_safe(symbol, interval, body.get("exchange"))
            else:
                return jsonify({"error": "Fonte de dados invalida"}), 400

        method = params.get("method", "hmm")
        n_states = int(params.get("n_states", 0))
        features = params.get("features", ["log_return", "volatility"])
        vol_window = int(params.get("vol_window", 20))
        causal = bool(params.get("causal", False))

        result = detect_regimes(
            df,
            method=method,
            n_states=n_states,
            features=features,
            vol_window=vol_window,
            causal=causal,
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  Trade Journal — registro manual de operações por estratégia
# ─────────────────────────────────────────────────────────────────────────────

JOURNAL_FILE = Path(__file__).parent / "data" / "journal_data.json"
_journal_lock = threading.Lock()


def _journal_load() -> dict:
    """Lê o journal do disco. Estrutura: {capital_inicial, trades:[...]}."""
    if not JOURNAL_FILE.exists():
        return {"capital_inicial": 0.0, "trades": []}
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("capital_inicial", 0.0)
        data.setdefault("trades", [])
        return data
    except Exception:
        return {"capital_inicial": 0.0, "trades": []}


def _journal_save(data: dict) -> None:
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _journal_stats(data: dict) -> dict:
    """Calcula métricas agregadas e por estratégia a partir dos trades."""
    trades = data.get("trades", [])
    capital_inicial = float(data.get("capital_inicial", 0.0) or 0.0)

    n = len(trades)
    wins = [t for t in trades if t.get("result") == "gain"]
    losses = [t for t in trades if t.get("result") == "loss"]
    gross_gain = sum(abs(float(t.get("amount", 0))) for t in wins)
    gross_loss = sum(abs(float(t.get("amount", 0))) for t in losses)
    net_pnl = gross_gain - gross_loss
    # Taxas reais pagas às corretoras (vindas do sync; manuais não têm fee).
    total_fees = sum(abs(float(t.get("fee", 0) or 0)) for t in trades)
    win_rate = (len(wins) / n * 100) if n else 0.0
    avg_win = (gross_gain / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    profit_factor = (gross_gain / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_gain > 0 else 0.0)
    expectancy = (net_pnl / n) if n else 0.0
    capital_atual = capital_inicial + net_pnl
    roi = (net_pnl / capital_inicial * 100) if capital_inicial > 0 else 0.0

    # Curva de capital ordenada por data (id como desempate)
    ordered = sorted(trades, key=lambda t: (t.get("date", ""), t.get("id", 0)))
    equity = []
    running = capital_inicial
    for t in ordered:
        amt = abs(float(t.get("amount", 0)))
        running += amt if t.get("result") == "gain" else -amt
        equity.append({"date": t.get("date", ""), "capital": round(running, 2)})

    # Agregado por estratégia
    by_strat: dict = {}
    for t in trades:
        s = t.get("strategy", "—") or "—"
        amt = abs(float(t.get("amount", 0)))
        signed = amt if t.get("result") == "gain" else -amt
        b = by_strat.setdefault(s, {
            "strategy": s, "trades": 0, "wins": 0, "losses": 0,
            "gross_gain": 0.0, "gross_loss": 0.0, "net": 0.0, "fees": 0.0,
        })
        b["trades"] += 1
        b["fees"] += abs(float(t.get("fee", 0) or 0))
        if t.get("result") == "gain":
            b["wins"] += 1
            b["gross_gain"] += amt
        else:
            b["losses"] += 1
            b["gross_loss"] += amt
        b["net"] += signed
    for b in by_strat.values():
        b["win_rate"] = (b["wins"] / b["trades"] * 100) if b["trades"] else 0.0
        pf = (b["gross_gain"] / b["gross_loss"]) if b["gross_loss"] > 0 else (
            float("inf") if b["gross_gain"] > 0 else 0.0)
        b["profit_factor"] = None if pf == float("inf") else round(pf, 2)
        for k in ("gross_gain", "gross_loss", "net", "win_rate", "fees"):
            b[k] = round(b[k], 2)

    def _r(x):
        return None if x == float("inf") else round(x, 2)

    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "gross_gain": round(gross_gain, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "total_fees": round(total_fees, 2),
        "net_after_fees": round(net_pnl - total_fees, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": _r(profit_factor),
        "expectancy": round(expectancy, 2),
        "capital_inicial": round(capital_inicial, 2),
        "capital_atual": round(capital_atual, 2),
        "roi": round(roi, 2),
        "equity_curve": equity,
        "by_strategy": sorted(by_strat.values(), key=lambda b: b["net"], reverse=True),
    }


@app.route("/api/journal", methods=["GET"])
def api_journal_get():
    with _journal_lock:
        data = _journal_load()
    trades = sorted(data["trades"], key=lambda t: (t.get("date", ""), t.get("id", 0)),
                    reverse=True)
    return jsonify({
        "capital_inicial": data.get("capital_inicial", 0.0),
        "trades": trades,
        "stats": _journal_stats(data),
    })


@app.route("/api/journal/capital", methods=["POST"])
def api_journal_capital():
    body = request.get_json(force=True) or {}
    try:
        capital = float(body.get("capital_inicial", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "capital_inicial inválido"}), 400
    with _journal_lock:
        data = _journal_load()
        data["capital_inicial"] = capital
        _journal_save(data)
        stats = _journal_stats(data)
    return jsonify({"capital_inicial": capital, "stats": stats})


@app.route("/api/journal/trade", methods=["POST"])
def api_journal_add():
    body = request.get_json(force=True) or {}
    result = body.get("result")
    if result not in ("gain", "loss"):
        return jsonify({"error": "result deve ser 'gain' ou 'loss'"}), 400
    try:
        amount = abs(float(body.get("amount", 0)))
    except (TypeError, ValueError):
        return jsonify({"error": "amount inválido"}), 400
    with _journal_lock:
        data = _journal_load()
        next_id = max((t.get("id", 0) for t in data["trades"]), default=0) + 1
        trade = {
            "id": next_id,
            "date": body.get("date") or "",
            "strategy": (body.get("strategy") or "").strip() or "—",
            "asset": (body.get("asset") or "").strip(),
            "result": result,
            "amount": round(amount, 2),
            "notes": (body.get("notes") or "").strip(),
        }
        data["trades"].append(trade)
        _journal_save(data)
        stats = _journal_stats(data)
    return jsonify({"trade": trade, "stats": stats})


@app.route("/api/journal/trade/<int:trade_id>", methods=["PUT"])
def api_journal_update(trade_id):
    body = request.get_json(force=True) or {}
    with _journal_lock:
        data = _journal_load()
        trade = next((t for t in data["trades"] if t.get("id") == trade_id), None)
        if trade is None:
            return jsonify({"error": "trade não encontrado"}), 404
        if "result" in body:
            if body["result"] not in ("gain", "loss"):
                return jsonify({"error": "result inválido"}), 400
            trade["result"] = body["result"]
        if "amount" in body:
            try:
                trade["amount"] = round(abs(float(body["amount"])), 2)
            except (TypeError, ValueError):
                return jsonify({"error": "amount inválido"}), 400
        for k in ("date", "strategy", "asset", "notes"):
            if k in body:
                trade[k] = body[k].strip() if isinstance(body[k], str) else body[k]
        _journal_save(data)
        stats = _journal_stats(data)
    return jsonify({"trade": trade, "stats": stats})


@app.route("/api/journal/trade/<int:trade_id>", methods=["DELETE"])
def api_journal_delete(trade_id):
    with _journal_lock:
        data = _journal_load()
        before = len(data["trades"])
        data["trades"] = [t for t in data["trades"] if t.get("id") != trade_id]
        if len(data["trades"]) == before:
            return jsonify({"error": "trade não encontrado"}), 404
        _journal_save(data)
        stats = _journal_stats(data)
    return jsonify({"deleted": trade_id, "stats": stats})


@app.route("/api/journal/sync", methods=["POST"])
def api_journal_sync():
    """
    Sincroniza operações e taxas reais das corretoras (BingX, OKX, Hyperliquid)
    para o journal. Dedup por external_id: trades já importados são atualizados,
    novos são inseridos. Entradas manuais (sem external_id) são preservadas.

    Body opcional: { "exchanges": ["bingx","okx"], "since_days": 30 }
    """
    body = request.get_json(force=True) or {}
    exchanges = body.get("exchanges")
    since_days = int(body.get("since_days", 30) or 30)

    try:
        from providers.exchange_sync import sync_all
    except Exception as e:
        return jsonify({"error": f"módulo de sync indisponível: {e}"}), 500

    result = sync_all(exchanges=exchanges, since_days=since_days)
    imported = result["trades"]

    with _journal_lock:
        data = _journal_load()
        existing = data["trades"]
        by_ext = {t.get("external_id"): t for t in existing if t.get("external_id")}
        next_id = max((t.get("id", 0) for t in existing), default=0) + 1

        added, updated = 0, 0
        for it in imported:
            ext = it["external_id"]
            if ext in by_ext:
                # Atualiza campos vindos da corretora, preserva id e notas manuais.
                tgt = by_ext[ext]
                notes = tgt.get("notes", "")
                tgt.update(it)
                if notes:
                    tgt["notes"] = notes
                updated += 1
            else:
                it["id"] = next_id
                next_id += 1
                existing.append(it)
                by_ext[ext] = it
                added += 1

        _journal_save(data)
        stats = _journal_stats(data)
        trades = sorted(data["trades"],
                        key=lambda t: (t.get("date", ""), t.get("id", 0)), reverse=True)

    return jsonify({
        "added": added,
        "updated": updated,
        "by_exchange": result["by_exchange"],
        "warnings": result["warnings"],
        "trades": trades,
        "stats": stats,
    })


# ─── Automação (paper / Bybit demo) ──────────────────────────────────────
from automation.api import automation_bp  # noqa: E402
app.register_blueprint(automation_bp)

# ─── Terminal (monitor / screener / DES / alertas / notícias) ─────────────
from api.terminal_api import terminal_bp  # noqa: E402
app.register_blueprint(terminal_bp)

# ─── Agentes de IA (padrões do AgentHUB: loop nativo + gateways opcionais) ─
from api.agents_api import agents_bp  # noqa: E402
app.register_blueprint(agents_bp)

# ─── HFT on-chain (memecoins, paper) ───────────────────────────────────────
from api.hft_engine import hft_bp  # noqa: E402
app.register_blueprint(hft_bp)


def _start_automation_runner():
    """Religa o runner se havia deployments rodando quando o server caiu.
    Com debug=True o reloader duplica o processo — só inicia no filho real
    (WERKZEUG_RUN_MAIN); sem debug, inicia direto."""
    import os as _os
    if _os.environ.get("WERKZEUG_RUN_MAIN") != "true" and _os.environ.get("FLASK_DEBUG", "1") != "0":
        return
    from automation import store as _astore
    from automation.runner import ensure_started as _ensure
    _astore.init_db()
    if _astore.list_deployments(status="running"):
        _ensure()
# ═══════════════════════════════════════════════════════════════════════════
# DEGEN — memecoins on-chain (GeckoTerminal)
# ═══════════════════════════════════════════════════════════════════════════

import time as _time
import requests as _requests

DEGEN_CHAINS = {
    "robinhood": "Robinhood Chain",
    "solana": "Solana",
    "base": "Base",
    "eth": "Ethereum",
    "bsc": "BNB Chain",
}

_degen_cache = {}
_degen_lock = threading.Lock()
DEGEN_CACHE_TTL = 60  # segundos


def _degen_parse_pools(payload):
    """Converte resposta da GeckoTerminal (pools + included tokens) em lista simples."""
    included = {
        item["id"]: item["attributes"]
        for item in payload.get("included", [])
        if item.get("type") == "token"
    }
    tokens = []
    for pool in payload.get("data", []):
        a = pool.get("attributes", {})
        rel = pool.get("relationships", {})
        base_id = rel.get("base_token", {}).get("data", {}).get("id", "")
        base = included.get(base_id, {})
        txns = a.get("transactions", {}).get("h24", {}) or {}

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        pc = a.get("price_change_percentage", {}) or {}
        tokens.append({
            "pool_name": a.get("name"),
            "pool_address": a.get("address"),
            "symbol": base.get("symbol"),
            "name": base.get("name"),
            "token_address": base.get("address"),
            "image_url": base.get("image_url"),
            "price_usd": _f(a.get("base_token_price_usd")),
            "change_m5": _f(pc.get("m5")),
            "change_h1": _f(pc.get("h1")),
            "change_h6": _f(pc.get("h6")),
            "change_h24": _f(pc.get("h24")),
            "volume_h24": _f(a.get("volume_usd", {}).get("h24")),
            "fdv_usd": _f(a.get("fdv_usd")),
            "market_cap_usd": _f(a.get("market_cap_usd")),
            "liquidity_usd": _f(a.get("reserve_in_usd")),
            "buys_h24": txns.get("buys"),
            "sells_h24": txns.get("sells"),
            "created_at": a.get("pool_created_at"),
        })
    return tokens


@app.route("/api/degen/chains", methods=["GET"])
def degen_chains():
    return jsonify({"chains": DEGEN_CHAINS})


@app.route("/api/degen/tokens", methods=["GET"])
def degen_tokens():
    chain = request.args.get("chain", "robinhood")
    kind = request.args.get("kind", "trending")  # trending | new
    if chain not in DEGEN_CHAINS:
        return jsonify({"error": f"Chain inválida: {chain}"}), 400
    if kind not in ("trending", "new"):
        return jsonify({"error": f"Tipo inválido: {kind}"}), 400

    cache_key = (chain, kind)
    now = _time.time()
    with _degen_lock:
        cached = _degen_cache.get(cache_key)
        if cached and now - cached[0] < DEGEN_CACHE_TTL:
            return jsonify(cached[1])

    endpoint = "trending_pools" if kind == "trending" else "new_pools"
    url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/{endpoint}"
    try:
        resp = _requests.get(
            url,
            params={"include": "base_token", "page": 1},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        tokens = _degen_parse_pools(resp.json())
    except Exception as e:
        return jsonify({"error": f"Falha ao consultar GeckoTerminal: {e}"}), 502

    result = {
        "chain": chain,
        "chain_label": DEGEN_CHAINS[chain],
        "kind": kind,
        "updated_at": now,
        "tokens": tokens,
    }
    with _degen_lock:
        _degen_cache[cache_key] = (now, result)
    return jsonify(result)


# ─── Degen: Hype Radar (Twitter/X + on-chain) ────────────────────────────────

import os as _os
import re as _re
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(Path(__file__).parent / ".env")

DEGEN_DEX_SLUGS = {"eth": "ethereum"}  # GeckoTerminal slug -> DexScreener slug
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacyredirect.com",
    "https://xcancel.com",
]
_hype_cache = {}
HYPE_CACHE_TTL = 180


from providers import x_scraper as _x_scraper
_x_lock = threading.Lock()  # um navegador por vez


def _hype_twitter_playwright(query):
    """Busca real no X via navegador logado (x_scraper.py). Lento (~10s)."""
    if not _x_scraper.has_session():
        return None
    with _x_lock:
        return _x_scraper.search(query)


def _hype_twitter_apiio(query):
    """Busca tweets via twitterapi.io (requer TWITTERAPI_IO_KEY no .env)."""
    key = _os.getenv("TWITTERAPI_IO_KEY")
    if not key:
        return None
    try:
        resp = _requests.get(
            "https://api.twitterapi.io/twitter/tweet/advanced_search",
            params={"query": query, "queryType": "Latest"},
            headers={"x-api-key": key},
            timeout=20,
        )
        resp.raise_for_status()
        out = []
        for tw in resp.json().get("tweets", [])[:30]:
            out.append({
                "text": tw.get("text", ""),
                "user": tw.get("author", {}).get("userName"),
                "followers": tw.get("author", {}).get("followers"),
                "likes": tw.get("likeCount", 0),
                "retweets": tw.get("retweetCount", 0),
                "replies": tw.get("replyCount", 0),
                "views": tw.get("viewCount"),
                "created_at": tw.get("createdAt"),
                "url": tw.get("url"),
            })
        return out
    except Exception:
        return None


def _hype_twitter_nitter(query):
    """Best-effort: raspa busca de tweets em instâncias Nitter públicas."""
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}
    for base in NITTER_INSTANCES:
        try:
            resp = _requests.get(
                f"{base}/search", params={"f": "tweets", "q": query},
                headers=headers, timeout=10,
            )
            if resp.status_code != 200 or "timeline-item" not in resp.text:
                continue
            html = resp.text
            items = html.split('class="timeline-item')[1:21]
            tweets = []
            for it in items:
                m_text = _re.search(r'class="tweet-content[^"]*"[^>]*>(.*?)</div>', it, _re.S)
                m_user = _re.search(r'class="username"[^>]*>@?([\w]+)', it)
                m_date = _re.search(r'class="tweet-date"><a[^>]*title="([^"]+)"', it)
                stats = _re.findall(r'class="tweet-stat"[^>]*>.*?</span>\s*([\d,\.]*)', it, _re.S)
                text = _re.sub(r"<[^>]+>", "", m_text.group(1)).strip() if m_text else ""
                if not text:
                    continue
                nums = [int(s.replace(",", "")) for s in stats if s.strip().isdigit()]
                tweets.append({
                    "text": text,
                    "user": m_user.group(1) if m_user else None,
                    "followers": None,
                    "likes": nums[-1] if nums else 0,
                    "retweets": nums[-2] if len(nums) > 1 else 0,
                    "replies": nums[0] if nums else 0,
                    "views": None,
                    "created_at": m_date.group(1) if m_date else None,
                    "url": None,
                })
            if tweets:
                return tweets
        except Exception:
            continue
    return None


def _hype_social_bluesky(query):
    """Busca posts no Bluesky (grátis, sem key). api.bsky.app funciona nesta rede."""
    try:
        resp = _requests.get(
            "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={"q": query, "sort": "latest", "limit": 25},
            headers={"Accept": "application/json"},
            timeout=12,
        )
        resp.raise_for_status()
        out = []
        for p in resp.json().get("posts", []):
            rec = p.get("record", {})
            handle = p.get("author", {}).get("handle", "")
            rkey = p.get("uri", "").rsplit("/", 1)[-1]
            out.append({
                "text": rec.get("text", ""),
                "user": handle,
                "followers": None,
                "likes": p.get("likeCount", 0),
                "retweets": p.get("repostCount", 0),
                "replies": p.get("replyCount", 0),
                "views": None,
                "created_at": (rec.get("createdAt") or "")[:16].replace("T", " "),
                "url": f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else None,
            })
        return out if out else None
    except Exception:
        return None


def _hype_dexscreener(chain, token_address):
    """Puxa pares, socials e boosts do DexScreener para o token."""
    slug = DEGEN_DEX_SLUGS.get(chain, chain)
    try:
        resp = _requests.get(
            f"https://api.dexscreener.com/tokens/v1/{slug}/{token_address}",
            timeout=15,
        )
        resp.raise_for_status()
        pairs = resp.json()
        if not pairs:
            return None
        # Par mais líquido representa o token
        best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
        info = best.get("info") or {}
        return {
            "socials": info.get("socials") or [],
            "websites": info.get("websites") or [],
            "boosts": (best.get("boosts") or {}).get("active", 0),
            "txns": best.get("txns") or {},
            "volume": best.get("volume") or {},
            "liquidity": (best.get("liquidity") or {}).get("usd"),
            "market_cap": best.get("marketCap"),
            "price_change": best.get("priceChange") or {},
            "pair_created_at": best.get("pairCreatedAt"),
            "url": best.get("url"),
        }
    except Exception:
        return None


def _hype_score(dex, tweets):
    """Score 0-100: pressão compradora, turnover, aceleração e sinal social."""
    breakdown = {}

    txns = (dex or {}).get("txns", {})
    h24 = txns.get("h24", {}) or {}
    buys, sells = h24.get("buys") or 0, h24.get("sells") or 0
    total = buys + sells
    buy_ratio = buys / total if total else 0.5
    # 50/50 = neutro; >65% compras = forte
    breakdown["pressao_compradora"] = round(max(0.0, min(1.0, (buy_ratio - 0.35) / 0.35)) * 25, 1)

    vol = (dex or {}).get("volume", {})
    vol24 = vol.get("h24") or 0
    liq = (dex or {}).get("liquidity") or 0
    turnover = vol24 / liq if liq else 0
    # turnover 5x+ da liquidez em 24h = degen total
    breakdown["turnover"] = round(min(1.0, turnover / 5.0) * 25, 1)

    vol6 = vol.get("h6") or 0
    accel = (vol6 * 4 / vol24) if vol24 else 0
    # >1 = últimas 6h acima do ritmo das 24h
    breakdown["aceleracao"] = round(max(0.0, min(1.0, (accel - 0.5) / 1.5)) * 25, 1)

    social = 0.0
    if tweets is not None:
        n = len(tweets)
        eng = sum((t["likes"] or 0) + (t["retweets"] or 0) * 2 for t in tweets)
        social = min(1.0, n / 20) * 0.6 + min(1.0, eng / 2000) * 0.4
    else:
        socials = (dex or {}).get("socials", [])
        kinds = {s.get("type") or s.get("label", "").lower() for s in socials}
        social += 0.25 if any("twitter" in str(k).lower() or "x" == str(k).lower() for k in kinds) else 0
        social += 0.15 if any("telegram" in str(k).lower() for k in kinds) else 0
        social += 0.10 if (dex or {}).get("websites") else 0
        social += min(0.5, ((dex or {}).get("boosts") or 0) / 20)
    breakdown["social"] = round(min(1.0, social) * 25, 1)

    score = round(sum(breakdown.values()))
    if score >= 75:
        label = "HYPE REAL"
    elif score >= 55:
        label = "Aquecendo"
    elif score >= 35:
        label = "Morno"
    else:
        label = "Morto"
    return score, label, breakdown


@app.route("/api/degen/hype", methods=["GET"])
def degen_hype():
    chain = request.args.get("chain", "robinhood")
    token = request.args.get("token", "")
    symbol = request.args.get("symbol", "")
    if not token or chain not in DEGEN_CHAINS:
        return jsonify({"error": "Parâmetros: chain + token (address) obrigatórios"}), 400

    cache_key = (chain, token)
    now = _time.time()
    with _degen_lock:
        cached = _hype_cache.get(cache_key)
        if cached and now - cached[0] < HYPE_CACHE_TTL:
            return jsonify(cached[1])

    dex = _hype_dexscreener(chain, token)

    # Handle do X anunciado pelo próprio token (se houver) entra na query
    query = f"${symbol}" if symbol else token
    tweets = _hype_twitter_playwright(query)
    twitter_source = "x" if tweets is not None else None
    if tweets is None:
        tweets = _hype_twitter_apiio(query)
        twitter_source = "twitterapi.io" if tweets is not None else None
    if tweets is None:
        tweets = _hype_social_bluesky(query)
        twitter_source = "bluesky" if tweets is not None else None
    if tweets is None:
        tweets = _hype_twitter_nitter(query)
        twitter_source = "nitter" if tweets is not None else "unavailable"

    score, label, breakdown = _hype_score(dex, tweets)

    result = {
        "chain": chain,
        "token": token,
        "symbol": symbol,
        "score": score,
        "label": label,
        "breakdown": breakdown,
        "twitter_source": twitter_source,
        "tweets": tweets or [],
        "socials": (dex or {}).get("socials", []),
        "websites": (dex or {}).get("websites", []),
        "boosts": (dex or {}).get("boosts", 0),
        "dexscreener_url": (dex or {}).get("url"),
        "updated_at": now,
    }
    with _degen_lock:
        _hype_cache[cache_key] = (now, result)
    return jsonify(result)


if __name__ == "__main__":
    print("========================================")
    print("   Backtesting API - Flask Server")
    print("   http://localhost:5000")
    print("========================================")
    _start_automation_runner()
    app.run(debug=True, port=5000, host="0.0.0.0", threaded=True)
