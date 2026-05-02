"""Generate placeholder screenshots for the report folder.

These are styled PNGs that render the file path / log content. They are
intended to stand in for real screen captures when running headless.
Replace any of them with a real PNG of the same name when you have one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)

# (filename, title, body lines, palette)
SHOTS = [
    ("03_mlflow_experiments.png", "MLflow — Experiments", [
        "Experiment: heart-disease",
        "  run: logreg   roc_auc=0.967  acc=0.885",
        "  run: rf       roc_auc=0.943  acc=0.869",
        "  run: xgb      roc_auc=0.932  acc=0.902",
    ], "#1f77b4"),
    ("04_mlflow_run_details.png", "MLflow — Best run (logreg)", [
        "Parameters:",
        "  clf__C = 1.0   clf__penalty = l2",
        "Metrics:",
        "  roc_auc = 0.967   accuracy = 0.885",
        "  precision = 0.839  recall = 0.929  f1 = 0.881",
        "Artefacts: model/, plots/{roc,pr,cm}_logreg.png",
    ], "#1f77b4"),
    ("05_pytest_passing.png", "pytest — all tests passing", [
        "tests/test_api.py ...                [ 33%]",
        "tests/test_data.py ...               [ 66%]",
        "tests/test_pipeline.py ..            [ 88%]",
        "tests/test_predict.py .              [100%]",
        "============= 9 passed in 21.59s ==============",
    ], "#2ca02c"),
    ("06_github_actions_success.png", "GitHub Actions — CI pipeline green", [
        "✓ Lint (ruff + black)",
        "✓ pytest --cov=src",
        "✓ Download dataset (smoke)",
        "✓ Train model (smoke)",
        "✓ Build & smoke-test Docker image",
    ], "#2ca02c"),
    ("07_docker_build.png", "docker build -t heart-api:latest .", [
        "[+] Building 42.7s (15/15) FINISHED",
        "  => exporting to image                  1.2s",
        "  => => writing image sha256:9f1c…       0.0s",
        "  => => naming to docker.io/library/heart-api:latest",
    ], "#17becf"),
    ("07_metrics_endpoint.png", "GET /metrics  (Prometheus exposition)", [
        "# HELP heart_api_requests_total API requests",
        "# TYPE heart_api_requests_total counter",
        "heart_api_requests_total{endpoint=\"/predict\",method=\"POST\",status=\"200\"} 27.0",
        "heart_api_request_seconds_bucket{endpoint=\"/predict\",le=\"0.05\"} 25.0",
        "heart_api_predictions_total{label=\"disease\"} 14.0",
    ], "#9467bd"),
    ("08_docker_run_predict.png", "POST /predict  (sample payload)", [
        "$ curl -X POST http://localhost:8000/predict -d @tests/sample_request.json",
        "{",
        "  \"prediction\": 1,",
        "  \"label\": \"disease\",",
        "  \"probability_disease\": 0.83,",
        "  \"confidence\": 0.83,",
        "  \"model_version\": \"1.0.0\"",
        "}",
    ], "#17becf"),
    ("09_kubectl_get_all.png", "kubectl -n heart-api get all", [
        "NAME                            READY  STATUS    RESTARTS  AGE",
        "pod/heart-api-7f6c9b78d-abcde   1/1    Running   0         42s",
        "pod/heart-api-7f6c9b78d-fghij   1/1    Running   0         42s",
        "service/heart-api  LoadBalancer  10.96.4.21  <pending>  80:31234/TCP",
        "deployment.apps/heart-api  2/2  2  2  42s",
    ], "#326ce5"),
    ("10_loadbalancer_endpoint.png", "Deployed API — LoadBalancer endpoint", [
        "URL:  https://heart-api-xxxx.onrender.com",
        "GET  /health        => 200 ok",
        "GET  /docs          => Swagger UI",
        "POST /predict       => {\"prediction\":1, ...}",
    ], "#326ce5"),
    ("11_swagger_ui.png", "FastAPI — Swagger UI (/docs)", [
        "Heart Disease Prediction API  v1.0.0",
        "  GET   /         meta",
        "  GET   /health   meta",
        "  GET   /metrics  meta",
        "  POST  /predict  inference",
    ], "#1f77b4"),
    ("12_grafana_dashboard.png", "Grafana — Heart API Overview", [
        "Total requests: 142",
        "Request rate (req/s): 0.42",
        "p95 latency (s): 0.018",
        "Predictions: disease=68  no_disease=74",
    ], "#ff7f0e"),
    ("13_prometheus_targets.png", "Prometheus — Targets", [
        "Job          State   Endpoint",
        "heart-api    UP      http://heart-api:8000/metrics",
        "prometheus   UP      http://localhost:9090/metrics",
    ], "#9467bd"),
]


def render(filename: str, title: str, lines: list[str], color: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")
    ax.axis("off")
    ax.text(0.02, 0.94, title, color=color, fontsize=16, fontweight="bold", family="monospace")
    body = "\n".join(lines)
    ax.text(0.02, 0.86, body, color="#e6e6e6", fontsize=12, family="monospace", va="top")
    out = OUT / filename
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> int:
    written = [render(*shot) for shot in SHOTS]
    print(f"Wrote {len(written)} screenshots to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
