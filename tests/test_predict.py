"""Inference smoke test: trains a tiny model on the fly and predicts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.pipeline import build_preprocessor


def _toy_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(size=n) for f in NUMERIC_FEATURES}
    for f in CATEGORICAL_FEATURES:
        data[f] = rng.integers(0, 3, size=n)
    X = pd.DataFrame(data)
    y = (X["age"] + X["chol"] > 0).astype(int)
    return X, y


def test_pipeline_trains_and_predicts_two_classes():
    X, y = _toy_dataset()
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)
    assert preds.shape == (len(X),)
    assert proba.shape == (len(X), 2)
    assert set(np.unique(preds)).issubset({0, 1})
