"""Clean and preprocess the raw Cleveland Heart Disease dataset.

* Replaces ``?`` placeholders with NaN
* Casts columns to numeric
* Imputes missing values (median)
* Binarizes the multi-class target ``num`` -> ``target`` (0 vs 1+)
* Writes ``data/processed/heart_cleveland_processed.csv``
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    PROCESSED_FILENAME,
    RAW_DIR,
    RAW_FILENAME,
    TARGET,
)
from src.data.download import download

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_raw(raw_path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV, downloading it on demand."""
    raw_path = raw_path or (RAW_DIR / RAW_FILENAME)
    if not raw_path.exists():
        download(dest=raw_path)
    df = pd.read_csv(raw_path)
    log.info("Loaded raw frame: shape=%s", df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning + binarization, returning a new DataFrame."""
    df = df.replace("?", np.nan).copy()

    # All columns must be numeric for downstream sklearn pipelines.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        log.info("Imputing %d missing values with column medians.", n_missing)
        df = df.fillna(df.median(numeric_only=True))

    # Binary classification target: 0 = no disease, 1 = disease (num >= 1).
    df[TARGET] = (df["num"] >= 1).astype(int)
    df = df.drop(columns=["num"])

    log.info(
        "Cleaned frame: shape=%s, target balance=%s",
        df.shape,
        df[TARGET].value_counts().to_dict(),
    )
    return df


def save_processed(df: pd.DataFrame, dest: Path | None = None) -> Path:
    dest = dest or (PROCESSED_DIR / PROCESSED_FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    log.info("Wrote processed dataset to %s", dest)
    return dest


def run() -> Path:
    """End-to-end: raw -> cleaned -> persisted CSV. Returns processed path."""
    df = clean(load_raw())
    return save_processed(df)


def main() -> int:
    try:
        path = run()
        log.info("OK | processed dataset at %s", path)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("Preprocess failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
