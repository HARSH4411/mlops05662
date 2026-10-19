# ---- Heart Disease Risk API ----
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# install deps first for layer caching
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# app code + trained model artefact
COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/model.json models/

RUN useradd --create-home --uid 1001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# one worker per container: Prometheus metrics live in-process; scale with replicas instead
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
