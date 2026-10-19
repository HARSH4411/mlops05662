"""FastAPI model-serving service.

Endpoints
  GET  /           service info
  GET  /health     liveness/readiness (503 if model not loaded)
  POST /predict    single patient -> prediction + confidence
  POST /predict/batch  list of patients
  GET  /metrics    Prometheus metrics
Every request is logged as one JSON line (request id, path, status, latency).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import Patient, Prediction
from heart.predict import load_model, predict_records

# ---------- structured logging ----------


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
log = logging.getLogger("heart.api")
log.handlers = [handler]
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
log.propagate = False

# ---------- Prometheus metrics ----------
REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions by class", ["label"])
RISK = Histogram("heart_api_risk_probability", "Distribution of predicted risk",
                 buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, meta = load_model(os.getenv("MODEL_PATH"))
        STATE.update(model=model, meta=meta)
        MODEL_LOADED.set(1)
        log.info("model loaded", extra={"extra_fields": {"model": meta.get("model_name")}})
    except FileNotFoundError as exc:
        MODEL_LOADED.set(0)
        log.error(str(exc))
    yield
    STATE.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            log.info("request", extra={"extra_fields": {
                "request_id": rid, "method": request.method, "path": path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _model():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return STATE["model"], STATE["meta"]


def _respond(patients: list[Patient], rid: str) -> list[Prediction]:
    model, meta = _model()
    results = predict_records([p.model_dump() for p in patients], model=model)
    out = []
    for r in results:
        PREDICTIONS.labels(r["label"]).inc()
        RISK.observe(r["risk_probability"])
        out.append(Prediction(**r, model_name=meta.get("model_name", "unknown"),
                              model_version=meta.get("trained_at", "unknown"), request_id=rid))
    log.info("prediction", extra={"extra_fields": {"request_id": rid, "n": len(out),
                                                   "labels": [o.label for o in out]}})
    return out


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health",
            "predict": "/predict", "metrics": "/metrics"}


@app.get("/health")
def health():
    _, meta = _model()
    return {"status": "ok", "model": meta.get("model_name"),
            "test_roc_auc": meta.get("metrics", {}).get("test_roc_auc")}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient, request: Request):
    return _respond([patient], request.state.request_id)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(patients: list[Patient], request: Request):
    if not patients or len(patients) > 500:
        raise HTTPException(status_code=422, detail="Send between 1 and 500 patients")
    return _respond(patients, request.state.request_id)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
