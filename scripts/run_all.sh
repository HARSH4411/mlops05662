#!/usr/bin/env bash
# One-shot end-to-end run for a fresh Linux VM (e.g. BITS Prayogshala).
#   cd ~/workspace && git clone https://github.com/HARSH4411/mlops05662.git
#   cd mlops05662 && bash scripts/run_all.sh
# Everything lives under ~/workspace so it survives VM stop.
set -euo pipefail
cd "$(dirname "$0")/.."
step() { printf '\n\033[1;34m==== %s ====\033[0m\n' "$*"; }

step "1. Python environment"
python3 --version
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
export PYTHONPATH=src:.

step "2. Data acquisition + cleaning"
python scripts/download_data.py

step "3. EDA (figures -> reports/figures)"
python scripts/eda.py | tail -5

step "4. Training + MLflow tracking"
python -m heart.train
cat models/model.json | head -25

step "5. Lint + unit tests"
flake8 && echo "flake8 OK"
pytest -v

step "6. Serve the model"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build -t heart-api:latest .
  docker rm -f heart-api >/dev/null 2>&1 || true
  docker run -d --name heart-api -p 8000:8000 heart-api:latest
else
  echo "Docker not available on this VM -> running the same API with uvicorn"
  pkill -f "uvicorn api.app:app" 2>/dev/null || true
  nohup uvicorn api.app:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
fi
for i in $(seq 1 30); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
echo "GET /health:";  curl -s localhost:8000/health; echo
echo "POST /predict:"; curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json; echo

step "7. Monitoring: traffic + Prometheus metrics"
python scripts/load_test.py -n 100
curl -s localhost:8000/metrics | grep -E "^heart_api_(requests_total|predictions_total)" | head -10

step "8. MLflow UI"
pkill -f "mlflow ui" 2>/dev/null || true
nohup mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
sleep 5

step "DONE"
echo "API docs : http://localhost:8000/docs"
echo "Metrics  : http://localhost:8000/metrics"
echo "MLflow   : http://localhost:5000"
echo "Logs     : tail -f api.log   (or: docker logs -f heart-api)"
