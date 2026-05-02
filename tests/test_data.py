"""Tests for data loading + cleaning."""
from __future__ import annotations

import io

import pandas as pd

from src.config import COLUMNS, TARGET
from src.data.preprocess import clean


def _toy_raw() -> pd.DataFrame:
    csv = (
        ",".join(COLUMNS) + "\n"
        + "63,1,3,145,233,1,0,150,0,2.3,0,0,1,0\n"
        + "67,1,4,160,286,0,2,108,1,1.5,1,3,3,2\n"
        + "37,1,3,130,250,0,0,187,0,3.5,0,0,2,0\n"
        + "41,0,2,130,?,0,1,172,0,1.4,2,0,2,1\n"
    )
    return pd.read_csv(io.StringIO(csv))


def test_clean_imputes_missing_and_binarizes_target():
    df = clean(_toy_raw())
    assert df.isna().sum().sum() == 0
    assert TARGET in df.columns and "num" not in df.columns
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_clean_preserves_row_count():
    raw = _toy_raw()
    df = clean(raw)
    assert len(df) == len(raw)


def test_clean_columns_are_numeric():
    df = clean(_toy_raw())
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), col
