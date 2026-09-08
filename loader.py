import pandas as pd
import io
from config import COLUMN_MAP, NUMERIC_COLUMNS, REQUIRED_COLUMNS, CSV_ENCODINGS

# Labels do CONFIG_SCHEMA (otimizador) -> header esperado pelo COLUMN_MAP
COLUMN_ALIASES = {
    "Tipo": "MA",
    "Modo de Saída": "Saída",
    "Banda Sup (%)": "Banda (%)",
    "Ângulo Alta": "Ângulo",
    "Sair no Flat (cinza)": "Flat",
    "Tipo de Stop": "Stop",
    "ATR Mult": "Stop Param",
}


def load_csv(file) -> pd.DataFrame:
    """Carrega o CSV de backtesting de um path ou file-like, já tipado e renomeado."""
    df = _read_with_fallback(file)
    df = _clean_columns(df)
    df = _normalize_aliases(df)
    _validate_columns(df)
    df = _rename_columns(df)
    df = _cast_numeric(df)
    return df


def _read_with_fallback(file) -> pd.DataFrame:
    """Aceita path ou file-like (ex.: UploadedFile do Streamlit)."""
    if hasattr(file, "read"):
        raw = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        return _parse_bytes(raw)

    with open(file, "rb") as f:
        raw = f.read()
    return _parse_bytes(raw)


def _parse_bytes(raw: bytes) -> pd.DataFrame:
    """Testa cada encoding e delimitador ate um parse com as colunas esperadas."""
    for encoding in CSV_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        for sep in (",", ";"):
            df = _read_from_header(text, sep)
            if df is not None:
                return df

    raise ValueError(
        "Nao foi possivel ler o CSV. Verifique o encoding e o formato do arquivo."
    )


def _read_from_header(text: str, sep: str, max_metadata_rows: int = 20) -> pd.DataFrame | None:
    """Le o CSV pulando ate N linhas de metadados antes do header real."""
    lines = text.splitlines()
    for skip in range(min(max_metadata_rows, len(lines))):
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[skip:])), sep=sep)
        except Exception:
            continue
        if len(df.columns) > 1 and _has_required_cols(df):
            return df
    return None


def _has_required_cols(df: pd.DataFrame) -> bool:
    """Header valido = pelo menos 2 das colunas-chave presentes."""
    cols = {str(c).strip() for c in df.columns}
    return len({"Retorno (%)", "Score", "Trades"} & cols) >= 2


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tira espacos do header e as colunas Unnamed que `;;` no fim da linha gera."""
    df.columns = df.columns.str.strip()
    unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _normalize_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns and new not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def _validate_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatorias ausentes no CSV: {', '.join(missing)}\n"
            f"Colunas encontradas: {', '.join(df.columns.tolist())}"
        )


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    return df


def _cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
