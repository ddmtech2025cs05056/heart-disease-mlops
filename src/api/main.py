"""FastAPI service exposing /predict, /health, /metrics."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pythonjsonlogger import jsonlogger

from src import __version__
from src.api.schemas import (
    HIGH_RISK_EXAMPLE,
    LOW_RISK_EXAMPLE,
    HealthResponse,
    HeartFeatures,
    PredictionResponse,
)
from src.models.predict import load_model, predict_one

# --- structured JSON logging -------------------------------------------------
_handler = logging.StreamHandler()
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
log = logging.getLogger("heart-api")

# --- prometheus metrics ------------------------------------------------------
REQUESTS = Counter("heart_api_requests_total", "API requests", ["endpoint", "method", "status"])
LATENCY = Histogram("heart_api_request_seconds", "API request latency", ["endpoint"])
PREDICTIONS = Counter("heart_api_predictions_total", "Prediction outcomes", ["label"])


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    try:
        load_model()
        log.info("model_loaded_on_startup")
    except FileNotFoundError as exc:
        log.warning("startup_no_model", extra={"error": str(exc)})
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    version=__version__,
    description="Predicts heart-disease risk from patient features.",
    lifespan=lifespan,
)


_INSTRUMENTATION_EXCLUDED_PATHS = {"/metrics"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["x-request-id"] = rid
    if request.url.path in _INSTRUMENTATION_EXCLUDED_PATHS:
        return response
    REQUESTS.labels(request.url.path, request.method, str(response.status_code)).inc()
    LATENCY.labels(request.url.path).observe(elapsed)
    log.info(
        "request",
        extra={
            "request_id": rid,
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "duration_ms": round(elapsed * 1000, 2),
        },
    )
    return response


@app.get("/", tags=["meta"])
def root():
    return {"service": "heart-api", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    loaded = True
    try:
        load_model()
    except FileNotFoundError:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        version=__version__,
    )


@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    features: HeartFeatures = Body(
        ...,
        openapi_examples={
            "high_risk": {
                "summary": "High-risk patient (expects label='disease')",
                "description": "70yo male, chest pain type 4, BP 180, chol 320, exang=1, oldpeak 4.0, ca=3, thal=7.",
                "value": HIGH_RISK_EXAMPLE,
            },
            "low_risk": {
                "summary": "Low-risk patient (expects label='no_disease')",
                "description": "35yo female, chest pain type 1, BP 110, chol 180, no exang, oldpeak 0, ca=0, thal=3.",
                "value": LOW_RISK_EXAMPLE,
            },
        },
    ),
):
    try:
        result = predict_one(features.model_dump())
    except FileNotFoundError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    PREDICTIONS.labels(result["label"]).inc()
    return PredictionResponse(model_version=__version__, **result)
