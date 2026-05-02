"""Inference helpers used by both the FastAPI service and tests."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_FILENAME, MODELS_DIR

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(path: Path | None = None):
    """Load and cache the trained sklearn pipeline from disk."""
    path = path or (MODELS_DIR / MODEL_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {path}. Run `python -m src.models.train` first."
        )
    log.info("Loading model from %s", path)
    return joblib.load(path)


def predict_one(features: dict) -> dict:
    """Run a single-record prediction.

    Returns a dict with ``prediction`` (0/1), ``label``, and ``confidence``.
    """
    model = load_model()
    df = pd.DataFrame([features])
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "label": "disease" if pred == 1 else "no_disease",
        "probability_disease": proba,
        "confidence": proba if pred == 1 else 1.0 - proba,
    }
