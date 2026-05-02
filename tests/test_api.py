"""FastAPI route tests using the TestClient."""

from __future__ import annotations

from pathlib import Path

import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.main import app
from src.config import MODEL_FILENAME, MODELS_DIR
from src.features.pipeline import build_preprocessor
from src.models import predict as predict_mod
from tests.test_predict import _toy_dataset


def _ensure_model() -> Path:
    """Train a tiny pipeline and persist it so the API can serve it."""
    path = MODELS_DIR / MODEL_FILENAME
    X, y = _toy_dataset()
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    predict_mod.load_model.cache_clear()
    return path


def test_root_and_health():
    _ensure_model()
    client = TestClient(app)
    assert client.get("/").status_code == 200
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True


def test_predict_endpoint(sample_payload):
    _ensure_model()
    client = TestClient(app)
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("disease", "no_disease")


def test_metrics_endpoint_exposes_prometheus():
    _ensure_model()
    client = TestClient(app)
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"heart_api_requests_total" in r.content
