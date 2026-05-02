"""Train Logistic Regression / Random Forest / XGBoost on the Cleveland dataset.

Performs stratified 5-fold GridSearchCV per model, logs everything to MLflow,
saves the best pipeline as ``models/heart_pipeline.joblib`` and writes
``models/best_model.json`` with the headline metrics.

Run with: ``python -m src.models.train``
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_META,
    CV_FOLDS,
    MLFLOW_EXPERIMENT,
    MLRUNS_DIR,
    MODEL_FILENAME,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    PROCESSED_FILENAME,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data.preprocess import run as ensure_processed
from src.features.pipeline import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _candidates() -> dict[str, tuple[object, dict]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l2"]},
        ),
        "rf": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10]},
        ),
        "xgb": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
                verbosity=0,
            ),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def _save_plots(
    name: str, y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray
) -> list[Path]:
    paths: list[Path] = []
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title=f"ROC — {name}", xlabel="FPR", ylabel="TPR")
    ax.legend()
    p = PLOTS_DIR / f"roc_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    pr_p, pr_r, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pr_r, pr_p, label=f"AP={average_precision_score(y_true, y_proba):.3f}")
    ax.set(title=f"Precision-Recall — {name}", xlabel="Recall", ylabel="Precision")
    ax.legend()
    p = PLOTS_DIR / f"pr_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion — {name}")
    p = PLOTS_DIR / f"cm_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def train() -> dict:
    processed_csv = PROCESSED_DIR / PROCESSED_FILENAME
    if not processed_csv.exists():
        ensure_processed()
    df = pd.read_csv(processed_csv)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    leaderboard: list[dict] = []
    for name, (clf, grid) in _candidates().items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_
            y_pred = best.predict(X_te)
            y_proba = best.predict_proba(X_te)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred),
                "recall": recall_score(y_te, y_pred),
                "f1": f1_score(y_te, y_pred),
                "roc_auc": roc_auc_score(y_te, y_proba),
                "cv_best_roc_auc": gs.best_score_,
            }
            mlflow.log_params({"model": name, **gs.best_params_})
            mlflow.log_metrics(metrics)
            for p in _save_plots(name, y_te.values, y_proba, y_pred):
                mlflow.log_artifact(str(p), artifact_path="plots")
            mlflow.sklearn.log_model(best, artifact_path="model")
            leaderboard.append({"name": name, "estimator": best, "metrics": metrics})
            log.info("%s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})

    leaderboard.sort(key=lambda r: r["metrics"]["roc_auc"], reverse=True)
    winner = leaderboard[0]
    out_model = MODELS_DIR / MODEL_FILENAME
    joblib.dump(winner["estimator"], out_model)
    meta = {
        "best_model": winner["name"],
        "metrics": winner["metrics"],
        "artefact": str(out_model.name),
    }
    (MODELS_DIR / BEST_MODEL_META).write_text(json.dumps(meta, indent=2))
    log.info("Saved best model (%s) to %s", winner["name"], out_model)
    return meta


def main() -> int:
    try:
        meta = train()
        print(json.dumps(meta, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
