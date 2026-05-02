"""Centralized project configuration paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = ROOT_DIR / "models"
PLOTS_DIR: Path = MODELS_DIR / "plots"

MLRUNS_DIR: Path = ROOT_DIR / "mlruns"

# UCI Heart Disease (Cleveland) dataset
UCI_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

RAW_FILENAME: str = "heart_cleveland_raw.csv"
PROCESSED_FILENAME: str = "heart_cleveland_processed.csv"

COLUMNS: list[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET: str = "target"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

MLFLOW_EXPERIMENT: str = "heart-disease"
MODEL_FILENAME: str = "heart_pipeline.joblib"
BEST_MODEL_META: str = "best_model.json"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
