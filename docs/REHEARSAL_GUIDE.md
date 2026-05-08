# Demo Video Rehearsal Guide

**Project:** End-to-End Heart Disease Prediction (MLOps Assignment 1)
**Author:** Deepak Dharmani — BITS ID 2025CS05056
**Repo:** https://github.com/dd-mtech123/heart-disease-mlops
**Live API:** https://heart-api-bxq0.onrender.com
**Target length:** 6 min 15 sec

---

## Pre-recording checklist (10 min before record)

- [ ] Close all browser tabs except the demo set
- [ ] Hide Windows taskbar (right-click → Settings → Auto-hide)
- [ ] Set display scaling to 100 %
- [ ] VS Code open at repo root; collapse `.venv/`, `.obs/`, `mlruns/`, `models/plots/`
- [ ] Hit `https://heart-api-bxq0.onrender.com/health` once (wakes Render)
- [ ] Confirm local stack: uvicorn `:8000`, Prometheus `:9090`, Grafana `:3000`
- [ ] Start MLflow UI in a separate PowerShell: `mlflow ui --port 5000`
- [ ] Pre-load these 8 tabs in **one** browser window, in order:
  1. https://github.com/dd-mtech123/heart-disease-mlops
  2. https://github.com/dd-mtech123/heart-disease-mlops/actions
  3. http://127.0.0.1:5000  *(MLflow UI)*
  4. http://127.0.0.1:8000/docs
  5. https://dashboard.render.com → heart-api page
  6. https://heart-api-bxq0.onrender.com/docs
  7. http://127.0.0.1:9090/targets
  8. http://127.0.0.1:3000/d/heart-api-overview
- [ ] Terminal pre-positioned at repo root, font size ≥ 16
- [ ] Recorder ready: **OBS Studio** (1080p / 30 fps / H.264 MP4)
- [ ] Use a **headset mic**, not laptop mic

---

## Scene 1 — Intro (0:00 – 0:25)

**Show:** GitHub repo home page; slowly scroll the README.

**Say:** "Hi, I'm Deepak Dharmani, BITS ID 2025CS05056. This is my end-to-end MLOps assignment for predicting heart disease on the UCI Cleveland dataset. The repository is on GitHub at `dd-mtech123/heart-disease-mlops`. In the next six minutes I'll walk through the full pipeline: data, modeling with *MLflow* tracking, tests and CI, the Dockerized FastAPI service, deployment to *Render.com*, and monitoring with *Prometheus and Grafana*."

---

## Scene 2 — Repo structure & EDA (0:25 – 1:00)

**Show:** VS Code file tree (`src/`, `tests/`, `deploy/`, `monitoring/`, `notebooks/`, `docs/`, `.github/workflows/`). Open `notebooks/01_eda.ipynb`; scroll past the class-balance plot and correlation heatmap.

**Say:** "The repo follows standard MLOps layout — `src/` for source, `tests/` for unit tests, `deploy/` for *Kubernetes manifests*, the *Helm chart*, and the *Render blueprint*, `monitoring/` for the *Prometheus config and Grafana dashboard*, and `.github/workflows/` for *CI*. The EDA notebook profiles the 303-row Cleveland dataset — class balance roughly 54% no-disease, 46% disease. The correlation heatmap shows `cp`, `thalach`, `oldpeak`, and `ca` as the strongest predictors."

---

## Scene 3 — Modeling + MLflow (1:00 – 2:00)

**Show:** `src/models/train.py` for ~5 sec → switch to MLflow UI (`http://127.0.0.1:5000`):
1. Click the **heart_disease** experiment
2. Show runs table (LogReg, RandomForest, XGBoost rows)
3. Sort by `roc_auc` desc — best run on top
4. Click best run → show **Parameters**, **Metrics**, **Artifacts** (joblib + ROC/PR/CM plots)

**Say:** "Training is in `src/models/train.py`. I trained three models — *Logistic Regression*, *Random Forest*, and *XGBoost* — each tuned with `GridSearchCV` over a stratified 5-fold cross-validation. Every run is logged to *MLflow*. Here's the experiment dashboard — runs, parameters, the six metrics tracked, and the artefacts: the joblib pipeline, ROC curve, precision-recall curve, and confusion matrix. The best model — sorted by ROC-AUC — was *Logistic Regression* with a CV-AUC of around 0.91. That joblib is what the API serves."

---

## Scene 4 — Tests + CI (2:00 – 2:45)

**Show:** PowerShell, run:

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Wait for green output → switch to GitHub Actions tab → green ✅ on latest run → click into it to show jobs.

**Say:** "Tests live in `tests/` — nine unit tests covering the data pipeline, the feature transformer, the predict function, and the FastAPI routes. They all pass locally in about 19 seconds. The same suite runs on every push via *GitHub Actions* — the workflow has been green for the last several commits. CI also runs `ruff` and `black` for linting, then a smoke training run that uploads the resulting model as a build artefact."

---

## Scene 5 — Containerized API (2:45 – 3:30)

**Show:** `Dockerfile` in VS Code, scroll slowly → switch to local Swagger (`http://127.0.0.1:8000/docs`). Expand `POST /predict` → **Try it out** → Examples dropdown → "Low-risk patient" → **Execute** (200, `label="no_disease"`). Switch dropdown to "High-risk patient" → **Execute** (200, `label="disease"`).

**Say:** "The model is served by a *FastAPI* application, containerised with a multi-stage *Dockerfile* — a `builder` stage for compiling dependencies and a slim runtime stage that runs as a non-root user with a built-in `HEALTHCHECK`. Here's the Swagger UI on my local instance. I've built two example payloads — high-risk and low-risk. The low-risk patient returns about 0.4% probability of disease. The high-risk patient — same schema, very different feature values — returns 99.7%. Both come back in well under a hundred milliseconds."

---

## Scene 6 — Production deployment on Render (3:30 – 4:30)

**Show:** `deploy/render/render.yaml` (5 sec) → Render dashboard tab (green "Live" badge, "Deploy live for 0b9262b" event) → public Swagger (`https://heart-api-bxq0.onrender.com/docs`). Repeat the high-risk Try-It-Out → 200 response.

**Say:** "For production deployment I have three artefacts in the repo: raw *Kubernetes* manifests under `deploy/k8s/` — Namespace, Deployment, LoadBalancer service, Ingress, and a Horizontal Pod Autoscaler — plus an equivalent *Helm chart* under `deploy/helm/heart-api/`, plus a *Render Blueprint* for one-click public deployment. I deployed via the Render Blueprint. This is the live service dashboard — the green badge confirms healthcheck-verified, and the Events log shows the build pulled commit `0b9262b`, ran the Dockerfile, and went live in about four minutes. And here's the **public** Swagger at `heart-api-bxq0.onrender.com/docs` — same API, same prediction, served over HTTPS by Render's load balancer. High-risk patient — 99.7% probability of disease."

---

## Scene 7 — Monitoring & Logging (4:30 – 5:45)

**7a (15 sec):** `http://127.0.0.1:8000/metrics` — scroll past `heart_api_requests_total`, `heart_api_request_seconds_bucket`, `heart_api_predictions_total`.

> "*Monitoring* is built on Prometheus. The API exposes a `/metrics` endpoint in Prometheus exposition format with three custom metric families: a *counter* for total requests labelled by endpoint, method, and status; a *histogram* for request latency; and a *counter* for predictions labelled by outcome."

**7b (25 sec):** `http://127.0.0.1:9090/targets` — both UP. Click **Graph**, paste:

```
sum by (endpoint) (rate(heart_api_requests_total[1m]))
```

Click **Execute**.

> "A local *Prometheus* server scrapes that endpoint every 5 seconds — both targets are healthy and green. From the Prometheus query interface I can run *PromQL* — here's request rate by endpoint over the last minute."

**7c (35 sec):** `http://127.0.0.1:3000/d/heart-api-overview` — show 4 panels. Hop to terminal:

```powershell
$env:PYTHONPATH=(Get-Location).Path; .venv\Scripts\python.exe -c "import json,requests; [requests.post('http://127.0.0.1:8000/predict', json=json.load(open('tests/sample_request.json'))) for _ in range(20)]"
```

Switch back to Grafana — request-rate panel spikes within 5 sec.

> "The Grafana dashboard is provisioned automatically from `monitoring/grafana/dashboards/heart-api.json`. Four panels: total requests, request rate by endpoint, p95 latency, and predictions by label. Watch what happens when I generate twenty live predictions … the request-rate panel spikes immediately and the predictions-by-label counter ticks up. Logging is structured *JSON* via `python-json-logger` — every request line includes a `request_id`, path, method, status, and `duration_ms`, ready to ship to ELK, Loki, or CloudWatch."

---

## Scene 8 — Wrap-up (5:45 – 6:15)

**Show:** Open `docs/REPORT.docx`; scroll the table of contents and deliverables checklist.

**Say:** "To summarise — the repo covers all nine assignment sections: data and EDA; three models tuned with cross-validation; *MLflow* experiment tracking; reproducible packaging via `requirements.txt`, Conda env, and joblib; a unit-tested codebase wired into *GitHub Actions*; a Dockerised *FastAPI* service; deployed publicly on *Render* with Kubernetes and Helm artefacts also provided; *Prometheus and Grafana* monitoring with structured JSON logs; and a ten-page report with nineteen embedded screenshots. The full code is at `github.com/dd-mtech123/heart-disease-mlops`, and the live API is at `heart-api-bxq0.onrender.com`. Thank you."

---

## One-page cheat sheet (keep beside keyboard)

```
[0:00] GitHub home               "Hi I'm Deepak…"
[0:25] VS Code tree + EDA nb     "Repo follows standard layout…"
[1:00] MLflow UI                 "Three models, MLflow tracks…"
[2:00] pytest -v + GH Actions    "Nine tests, CI green…"
[2:45] Dockerfile + local Swagger "FastAPI, multi-stage Docker…"
[3:30] render.yaml + Render UI + public Swagger  "Render Blueprint deploy…"
[4:30] /metrics → Prom targets → Grafana + traffic spike  "Monitoring…"
[5:45] REPORT.docx               "To summarise…"
[6:15] END
```

---

## Scoring tips

- Show ≥ 3 **live actions** (`pytest`, Try-It-Out, Grafana traffic spike)
- MLflow: show **multiple runs** + the **artifacts panel** with joblib visible
- CI: show a **green Actions run** with job logs expanded
- Deployment: type the **public URL** into the address bar live
- Monitoring: generate traffic *during* recording so panels visibly move
- Drop the keywords: "non-root user", "healthcheck", "structured logs", "autoscaler"

## Mistakes to avoid

- Reading monotonously → paraphrase, don't recite
- Showing desktop / file explorer / personal tabs
- Long silent gaps while pages load → narrate over them
- Ambiguity between local and deployed → always say "local instance" vs "**public** URL on Render"
- Spending 30 sec reading code → spend time *running* things, not *reading* them
- Recording at 720p → 1080p minimum; zoom into terminals if text is small
- Final video > 10 min → keep it tight
