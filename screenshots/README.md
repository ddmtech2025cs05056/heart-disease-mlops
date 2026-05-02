# Screenshots

The PNGs in this folder are referenced by `docs/REPORT.md` and
`docs/demo_slideshow.mp4`.

| File | Subject |
|---|---|
| `01_eda_target_distribution.png` | EDA — class balance |
| `02_eda_correlation_heatmap.png` | EDA — feature correlation |
| `03_mlflow_experiments.png` | MLflow — experiment leaderboard |
| `04_mlflow_run_details.png` | MLflow — best run details |
| `05_pytest_passing.png` | `pytest` — all tests passing |
| `06_github_actions_success.png` | GitHub Actions — CI green |
| `07_docker_build.png` | `docker build` output |
| `07_metrics_endpoint.png` | `GET /metrics` (Prometheus exposition) |
| `08_docker_run_predict.png` | `POST /predict` example |
| `09_kubectl_get_all.png` | `kubectl get all` output |
| `10_loadbalancer_endpoint.png` | Deployed API URL |
| `11_swagger_ui.png` | FastAPI Swagger UI |
| `12_grafana_dashboard.png` | Grafana dashboard |
| `13_prometheus_targets.png` | Prometheus targets |

To regenerate placeholders:

```bash
python scripts/capture_screenshots.py
```

Replace any PNG with a real screen capture of the same name to upgrade.
