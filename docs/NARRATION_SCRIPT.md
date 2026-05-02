# Demo video — narration script

Length target: ~90 s. Read alongside `docs/demo_slideshow.mp4`.

| t (s) | Scene | Narration |
|---|---|---|
| 00–05 | Title slide | "Heart Disease Prediction — an end-to-end MLOps project for BITS MLOps Assignment 1." |
| 05–15 | EDA | "We start with the UCI Cleveland dataset — 303 rows, 13 features, binary target. The class balance is roughly 54 / 46." |
| 15–25 | Correlation heatmap | "A correlation heatmap exposes the strongest single-feature signals: `cp`, `thalach`, `oldpeak`, and `ca`." |
| 25–40 | MLflow leaderboard + best run | "We run Logistic Regression, Random Forest, and XGBoost through 5-fold GridSearchCV. MLflow records every parameter, metric, and plot. Logistic Regression wins with ROC-AUC 0.963." |
| 40–50 | pytest output | "Unit tests cover the cleaning, the preprocessing pipeline, model inference, and the API — all 11 pass." |
| 50–60 | GitHub Actions green | "On every push, GitHub Actions lints, tests, retrains a smoke model, and builds + probes the Docker image." |
| 60–70 | Docker + /predict | "The Dockerfile is multi-stage and runs as a non-root user. A POST to `/predict` returns prediction, probability, and confidence." |
| 70–80 | Kubernetes + deployed URL | "We deploy with raw manifests *and* a Helm chart — `Deployment`, `LoadBalancer`, `Ingress`, and `HPA`. Public URL: <DEPLOYED_URL>." |
| 80–90 | Grafana / Prometheus | "Prometheus scrapes `/metrics`; Grafana plots request rate, p95 latency, and prediction counts in real time. That closes the loop — code → CI → container → cluster → observability." |
