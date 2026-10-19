# Heart Disease Risk — End-to-End MLOps Pipeline

BITS Pilani · MLOps (S1-26_AIMLZG523) · Assignment 1

A binary classifier that predicts the presence of heart disease from 13 clinical attributes
(UCI Heart Disease, Cleveland subset), taken all the way from raw data to a monitored,
containerised API on Kubernetes with CI/CD.

```
 UCI repo ──► download_data.py ──► clean CSV ──► eda.py (figures)
                                        │
                                        ▼
                           heart.train (GridSearchCV × 3 models)
                                        │  params / metrics / plots / model
                                        ├──────────────► MLflow (sqlite + artifacts, registry)
                                        ▼
                           models/model.joblib  (preprocessing + classifier pipeline)
                                        │
             GitHub Actions: lint ─► test ─► train ─► docker build + smoke test ─► GHCR
                                        │
                                        ▼
                 FastAPI container (/predict, /health, /metrics, JSON logs)
                                        │
             Kubernetes (Deployment ×2, LoadBalancer Service, Ingress) ◄── Prometheus ─► Grafana
```

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/download_data.py` | Downloads raw UCI file, cleans it, writes `data/processed/heart_clean.csv` |
| `scripts/eda.py` | EDA figures → `reports/figures/` |
| `notebooks/eda_training_inference.ipynb` | Interactive EDA, model comparison and inference walkthrough (executed) |
| `src/heart/` | Package: `config`, `data` (clean), `features` (ColumnTransformer), `train` (CV + MLflow), `predict` |
| `api/` | FastAPI app with request logging + Prometheus metrics |
| `tests/` | Pytest unit tests for data, features, models and API |
| `.github/workflows/ci.yml` | Lint → test → train → Docker build/smoke-test → push to GHCR |
| `Dockerfile`, `docker-compose.yml` | Serving image; local stack with Prometheus + Grafana |
| `k8s/` | Namespace, Deployment, Service (LoadBalancer), Ingress, in-cluster Prometheus |
| `monitoring/` | Prometheus scrape config, Grafana datasource + dashboard |
| `models/` | Final model pipeline (`model.joblib`) + metadata (`model.json`) |
| `screenshots/` | Screenshots for the report |
| `reports/MLOps_Assignment1_Report.docx` | Final written report |

## Quick start on a fresh Linux VM (e.g. BITS Prayogshala)

```bash
cd ~/workspace && git clone https://github.com/HARSH4411/mlops05662.git && cd mlops05662
bash scripts/run_all.sh
```

Runs every stage (data, EDA, training + MLflow, lint, tests, API, monitoring). Uses Docker if the VM has it, otherwise serves the same API with uvicorn.

## 1 · Setup (Windows PowerShell; Linux/macOS commands in brackets)

```powershell
git clone https://github.com/HARSH4411/mlops05662.git; cd mlops05662
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # [source .venv/bin/activate]
pip install -r requirements.txt
$env:PYTHONPATH="src;."                 # [export PYTHONPATH=src:.]
```

## 2 · Data + EDA

```powershell
python scripts/download_data.py         # raw -> data/processed/heart_clean.csv
python scripts/eda.py                   # figures -> reports/figures/
```

## 3 · Train + experiment tracking

```powershell
python -m heart.train                   # 3 models, 5-fold CV grid search, logged to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # http://localhost:5000
```

Best model (selected on mean CV ROC-AUC) is saved to `models/model.joblib` and registered
as `heart-disease-classifier` in the MLflow registry.

## 4 · Tests + lint

```powershell
flake8
pytest -v --cov=heart --cov=api
```

## 5 · Docker

```powershell
docker build -t heart-api:latest .
docker run -d --name heart-api -p 8000:8000 heart-api:latest
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@sample_request.json"
```

Swagger UI: http://localhost:8000/docs

Example response:
```json
{"prediction": 1, "label": "disease", "risk_probability": 0.9947, "confidence": 0.9947,
 "model_name": "logistic_regression", "model_version": "...", "request_id": "e9384c26cb6c"}
```

## 6 · Monitoring (Prometheus + Grafana)

```powershell
docker rm -f heart-api                  # free port 8000
docker compose up -d --build
python scripts/load_test.py -n 300      # generate traffic
```
Prometheus http://localhost:9090 · Grafana http://localhost:3001 (admin/admin) → *Heart API* dashboard.

## 7 · Kubernetes (Docker Desktop Kubernetes / Minikube)

```powershell
docker build -t heart-api:latest .
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/monitoring.yaml
kubectl -n heart rollout status deploy/heart-api
kubectl -n heart get pods,svc
curl.exe http://localhost:8080/health
curl.exe -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "@sample_request.json"
kubectl -n heart port-forward svc/prometheus 9090:9090     # in-cluster Prometheus
```
Minikube: `minikube image load heart-api:latest` then `minikube tunnel`.
Optional ingress: install ingress-nginx, `kubectl apply -f k8s/ingress.yaml`, map `heart.local` to 127.0.0.1.

## 8 · CI/CD

Every push / PR runs `.github/workflows/ci.yml`:
1. **lint** – flake8
2. **test** – pytest with coverage (JUnit + coverage XML uploaded as artifacts)
3. **train** – download data, EDA, train with MLflow, quality gate (test ROC-AUC ≥ 0.80), upload model + MLflow DB
4. **docker** – build image with the freshly trained model, run it, smoke-test `/health` and `/predict`, push to GHCR on `main`

Any failing step fails the pipeline.

## API contract

`POST /predict`

| field | type | notes |
|---|---|---|
| age | number | years |
| sex | 0/1 | 1 = male |
| cp | 1–4 | chest pain type |
| trestbps | number | resting BP (mm Hg) |
| chol | number | serum cholesterol (mg/dl) |
| fbs | 0/1 | fasting blood sugar > 120 |
| restecg | 0–2 | resting ECG |
| thalach | number | max heart rate |
| exang | 0/1 | exercise-induced angina |
| oldpeak | number | ST depression |
| slope | 1–3 | slope of peak ST segment |
| ca | 0–3 or null | vessels coloured by fluoroscopy |
| thal | 3/6/7 or null | thalassemia |

Invalid values return HTTP 422 with a validation message.
