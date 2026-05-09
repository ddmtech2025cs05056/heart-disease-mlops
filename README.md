# Heart Disease Prediction â€” End-to-End MLOps Project

[![CI](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/ci.yml)
[![Lint](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/lint.yml/badge.svg)](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/lint.yml)
[![CodeQL](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/codeql.yml/badge.svg)](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/codeql.yml)
[![Docker Build](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/docker-build.yml/badge.svg)](https://github.com/ddmtech2025cs05056/heart-disease-mlops/actions/workflows/docker-build.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🚀 Live deliverables

| Artefact | Link |
|---|---|
| 🎥 **Demo video (in repo)** | [`docs/2026-05-09 17-51-37.mp4`](docs/2026-05-09%2017-51-37.mp4) |
| 🎥 **Demo video (Google Drive — streaming)** | <https://drive.google.com/file/d/106ABOgXefeEAvOQdUCnteJ9P6vjYF_M1/view?usp=drive_link> |
| 📄 **Final report (PDF)** | [`docs/REPORT.pdf`](docs/REPORT.pdf) |
| 📝 **Final report (Word)** | [`docs/REPORT.docx`](docs/REPORT.docx) |
| 🌐 **Live API (Render)** | <https://heart-disease-mlops-yhb7.onrender.com> |
| 📘 **Try `/predict` (Swagger)** | <https://heart-disease-mlops-yhb7.onrender.com/docs#/inference/predict_predict_post> |
| ❤️ **Health check** | <https://heart-disease-mlops-yhb7.onrender.com/health> |

---

A reproducible, production-grade MLOps pipeline that predicts heart-disease
risk from patient health records using the **UCI Heart Disease (Cleveland)**
dataset. It covers EDA â†’ training â†’ MLflow tracking â†’ packaging â†’ CI/CD â†’
containerization â†’ Kubernetes/Render deployment â†’ Prometheus + Grafana
monitoring.

> Submitted for **MLOps (S2-25_AMLCSZG523) â€” Assignment 1**.

---

## 1. Deliverables

| # | Deliverable | Location |
|---|---|---|
| a | Source code, Dockerfile, requirements | this repository |
| a | Cleaned dataset + download script | `src/data/download.py`, `src/data/preprocess.py` |
| a | Notebooks (EDA + modelling) | `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb` |
| a | Unit tests | `tests/` |
| a | GitHub Actions workflow | `.github/workflows/ci.yml` |
| a | Deployment manifests + Helm chart | `deploy/k8s/`, `deploy/helm/heart-api/` |
| a | Screenshots | `screenshots/` |
| a | Final 10-page report | `docs/REPORT.pdf`, `docs/REPORT.docx`, `docs/REPORT.md` |
| b | Short demo video | `docs/2026-05-09 17-51-37.mp4` |
| c | Deployed API URL / local instructions | see Â§5 |

---

## 2. Architecture

```
+--------+   +----------------+   +-----------+   +------------+
|  UCI   |-->| Preprocessing  |-->|  Models   |-->|  MLflow    |
|  CSV   |   | (sklearn       |   | LR/RF/XGB |   |  Tracking  |
+--------+   |  Pipeline)     |   +-----+-----+   +------------+
             +----------------+         |
                                        v
                                 +-------------+
                                 |  joblib     |
                                 |  artefact   |
                                 +------+------+
                                        |
                +-----------+   +-------v--------+   +---------------+
                | GitHub    |-->|  Docker image  |-->|  Kubernetes   |
                | Actions   |   |  (FastAPI)     |   |  / Render     |
                +-----------+   +-------+--------+   +-------+-------+
                                        |                    |
                                        v                    v
                                 +--------------+    +----------------+
                                 |  Prometheus  |--->|    Grafana     |
                                 +--------------+    +----------------+
```

Higher-resolution diagram in `docs/architecture.md`.

---

## 3. Quickstart

```bash
git clone https://github.com/ddmtech2025cs05056/heart-disease-mlops.git
cd heart-disease-mlops

python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.download         # download UCI Cleveland CSV
python -m src.models.train          # train + log to MLflow + save joblib

uvicorn src.api.main:app --port 8000 --reload
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_request.json
```

---

## 4. Docker

```bash
docker build -t heart-api:latest .
docker run --rm -p 8000:8000 heart-api:latest
# Swagger: http://localhost:8000/docs
```

`docker compose up -d --build` brings up the API alongside **Prometheus**
(`:9090`) and **Grafana** (`:3000`, anonymous viewer enabled).

---

## 5. Deployment

### Option A â€” Public URL via Render.com (recommended)

1. Push this repo to GitHub.
2. On <https://dashboard.render.com> â†’ **New â†’ Blueprint** â†’ connect repo.
3. Render reads `deploy/render/render.yaml` and provisions a public
   Docker web service on the free tier with `/health` health-checks.
4. The public URL appears as `https://heart-api-xxxx.onrender.com`.
   Replace `<DEPLOYED_URL>` below and in `docs/REPORT.md`.

### Option B â€” Local Kubernetes (Minikube / Docker Desktop)

```bash
# Raw manifests
kubectl apply -f deploy/k8s/
kubectl -n heart-api get all

# OR Helm
helm upgrade --install heart-api deploy/helm/heart-api
kubectl port-forward svc/heart-api 8000:80
```

**Deployed API URL:** <https://heart-disease-mlops-yhb7.onrender.com>
*(Swagger: <https://heart-disease-mlops-yhb7.onrender.com/docs> · Health: <https://heart-disease-mlops-yhb7.onrender.com/health>)*

---

## 6. CI/CD

`.github/workflows/ci.yml` runs on every push / PR:

1. **Lint** â€” `ruff check` + `black --check`
2. **Test** â€” `pytest --cov=src`
3. **Train** â€” smoke train on the real UCI dataset, upload `models/` artefact
4. **Docker** â€” build the image, run it, probe `/health`

Pipeline fails fast on lint, test, train, or container errors.

---

## 7. Repository layout

```
.github/workflows/ci.yml
data/{raw,processed}/
notebooks/{01_eda,02_modeling}.ipynb
src/{config.py, data/, features/, models/, api/}
tests/test_{data,pipeline,predict,api}.py
deploy/{k8s/, helm/heart-api/, render/render.yaml}
monitoring/{prometheus.yml, grafana/}
docs/{REPORT.md, REPORT.docx, REPORT.pdf, architecture.md, 2026-05-09 17-51-37.mp4}
screenshots/*.png
Dockerfile, docker-compose.yml, Makefile, requirements.txt, environment.yml
```

---

## 8. License

MIT â€” see `LICENSE`.
