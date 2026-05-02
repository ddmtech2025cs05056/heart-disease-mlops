"""Generate the EDA and modelling notebooks programmatically.

Running this script writes:
    notebooks/01_eda.ipynb
    notebooks/02_modeling.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def _nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": src.splitlines(keepends=True),
    }


def write_eda() -> Path:
    cells = [
        md("# 01 — Exploratory Data Analysis\n\nUCI Heart Disease (Cleveland) — 303 rows × 14 cols."),
        code(
            "import sys, os\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import pandas as pd\n"
            "from src.data.preprocess import clean, load_raw\n"
            "from src.config import TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES\n"
            "sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1. Load + clean"),
        code("df = clean(load_raw())\nprint(df.shape)\ndf.head()"),
        md("## 2. Missing-value report"),
        code("df.isna().sum()"),
        md("## 3. Class balance"),
        code(
            "ax = df[TARGET].value_counts().plot(kind='bar', color=['#4C72B0','#DD8452'])\n"
            "ax.set(title='Target balance', xlabel='target (0=no, 1=disease)', ylabel='count')\n"
            "plt.tight_layout(); plt.savefig('../screenshots/01_eda_target_distribution.png', dpi=120)\n"
        ),
        md("## 4. Numeric distributions"),
        code(
            "df[NUMERIC_FEATURES].hist(bins=20, figsize=(10, 6), color='#4C72B0')\n"
            "plt.tight_layout()\n"
        ),
        md("## 5. Correlation heatmap"),
        code(
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)\n"
            "plt.title('Feature correlation')\n"
            "plt.tight_layout(); plt.savefig('../screenshots/02_eda_correlation_heatmap.png', dpi=120)\n"
        ),
        md("## 6. Categorical vs target"),
        code(
            "fig, axes = plt.subplots(2, 4, figsize=(14, 6))\n"
            "for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):\n"
            "    sns.countplot(data=df, x=col, hue=TARGET, ax=ax, palette='Set2')\n"
            "    ax.set_title(col)\n"
            "plt.tight_layout()\n"
        ),
    ]
    p = NB_DIR / "01_eda.ipynb"
    p.write_text(json.dumps(_nb(cells), indent=1), encoding="utf-8")
    return p


def write_modeling() -> Path:
    cells = [
        md("# 02 — Modelling\n\nLogReg / RandomForest / XGBoost via GridSearchCV (5-fold)."),
        code(
            "import sys, os\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "from src.models.train import train\n"
            "result = train()\n"
            "result\n"
        ),
        md("MLflow UI: `mlflow ui --backend-store-uri ./mlruns` then open <http://localhost:5000>."),
        md("## Inspect best model"),
        code(
            "import json, joblib\n"
            "from src.config import MODELS_DIR, MODEL_FILENAME, BEST_MODEL_META\n"
            "meta = json.loads((MODELS_DIR / BEST_MODEL_META).read_text())\n"
            "model = joblib.load(MODELS_DIR / MODEL_FILENAME)\n"
            "print(json.dumps(meta, indent=2))\n"
            "model\n"
        ),
    ]
    p = NB_DIR / "02_modeling.ipynb"
    p.write_text(json.dumps(_nb(cells), indent=1), encoding="utf-8")
    return p


def main() -> int:
    a = write_eda(); b = write_modeling()
    print("Wrote:", a, b); return 0


if __name__ == "__main__":
    raise SystemExit(main())
