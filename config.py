from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

TOP_N = 20

# Header do CSV (pt) -> nome interno (en)
COLUMN_MAP = {
    "Rank": "rank",
    "Data": "data",
    "Ativo": "ativo",
    "Timeframe": "timeframe",
    "Retorno (%)": "return_pct",
    "Max DD (%)": "max_dd_pct",
    "Trades": "trades",
    "Win Rate (%)": "win_rate_pct",
    "Avg Win (%)": "avg_win_pct",
    "Avg Loss (%)": "avg_loss_pct",
    "Profit Factor": "profit_factor",
    "Sharpe": "sharpe",
    "Sortino": "sortino",
    "Calmar": "calmar",
    "Omega": "omega",
    "Sterling": "sterling",
    "Burke": "burke",
    "Score": "score",
    "MA": "ma",
    "Período": "periodo",
    "Lookback": "lookback",
    "Ângulo": "angulo",
    "Saída": "saida",
    "Banda (%)": "banda_pct",
    "Alvo Fixo (%)": "alvo_fixo_pct",
    "Flat": "flat",
    "Stop": "stop",
    "Stop Param": "stop_param",
    "Pullback": "pullback",
    "Entry Zone": "entry_zone",
}

NUMERIC_COLUMNS = [
    "rank", "return_pct", "max_dd_pct", "trades", "win_rate_pct",
    "avg_win_pct", "avg_loss_pct", "profit_factor", "sharpe", "score",
    "lookback", "angulo", "banda_pct", "alvo_fixo_pct", "stop_param",
]

# Subset mínimo para os gráficos funcionarem
REQUIRED_COLUMNS = [
    "Retorno (%)", "Max DD (%)", "Trades", "Win Rate (%)",
    "Profit Factor", "Sharpe", "Score",
]

COLUMN_DISPLAY = {v: k for k, v in COLUMN_MAP.items()}

CSV_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
