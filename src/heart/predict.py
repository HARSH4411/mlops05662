"""Inference helpers shared by the API and the CLI."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart import config


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    p = Path(path or config.MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}. Run `python -m heart.train` first.")
    model = joblib.load(p)
    meta_p = p.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return model, meta


def predict_records(records: list[dict], model=None) -> list[dict]:
    if model is None:
        model, _ = load_model()
    X = pd.DataFrame.from_records(records, columns=config.FEATURES)
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= 0.5)
        out.append({
            "prediction": pred,
            "label": "disease" if pred else "no_disease",
            "risk_probability": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
        })
    return out


if __name__ == "__main__":
    sample = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
              "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
              "ca": 0, "thal": 6}
    print(predict_records([sample]))
