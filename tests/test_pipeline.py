"""Tests for the sklearn preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.features.pipeline import build_preprocessor


def _toy_X() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20
    data = {f: rng.normal(size=n) for f in NUMERIC_FEATURES}
    for f in CATEGORICAL_FEATURES:
        data[f] = rng.integers(0, 3, size=n)
    return pd.DataFrame(data)


def test_preprocessor_fits_and_transforms():
    pre = build_preprocessor()
    X = _toy_X()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)


def test_preprocessor_handles_unseen_categories():
    pre = build_preprocessor()
    X_train = _toy_X()
    pre.fit(X_train)
    X_new = _toy_X()
    X_new.loc[0, "cp"] = 99  # unseen
    out = pre.transform(X_new)
    assert out.shape[0] == len(X_new)
