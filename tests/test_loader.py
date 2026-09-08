import io

import pytest

from loader import load_csv

COLS = "Retorno (%),Max DD (%),Trades,Win Rate (%),Profit Factor,Sharpe,Score,Tipo"
ROW = "12.5,-3.1,40,55,1.8,1.2,9.1,EMA"


def _csv(text, encoding="utf-8"):
    return io.BytesIO(text.encode(encoding))


def test_reads_csv_and_renames_to_internal_columns():
    df = load_csv(_csv(f"{COLS}\n{ROW}"))
    assert df["trades"].iloc[0] == 40
    assert df["ma"].iloc[0] == "EMA"  # alias "Tipo" -> "MA" -> "ma"


def test_skips_metadata_rows_above_the_real_header():
    df = load_csv(_csv(f"Otimizacao BTC\ngerado em 2026-01-01\n\n{COLS}\n{ROW}"))
    assert df["score"].iloc[0] == 9.1


def test_reads_semicolon_separated_csv():
    df = load_csv(_csv(f"{COLS.replace(',', ';')}\n{ROW.replace(',', ';')}"))
    assert df["return_pct"].iloc[0] == 12.5


def test_reads_latin1_and_bom_encodings():
    assert load_csv(_csv(f"{COLS}\n{ROW}", encoding="latin-1"))["score"].iloc[0] == 9.1
    assert load_csv(_csv(f"﻿{COLS}\n{ROW}"))["score"].iloc[0] == 9.1


def test_non_numeric_cells_become_nan_not_errors():
    df = load_csv(_csv(f"{COLS}\n12.5,-3.1,n/d,55,1.8,1.2,9.1,EMA"))
    assert df["trades"].isna().iloc[0]


def test_rejects_csv_without_the_required_columns():
    with pytest.raises(ValueError):
        load_csv(_csv("a,b,c\n1,2,3"))
