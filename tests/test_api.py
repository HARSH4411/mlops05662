import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heart.data import clean, split_xy
from heart.predict import load_model
from heart.train import make_pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory, raw_df):
    X, y = split_xy(clean(raw_df))
    model = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump(model, path)
    path.with_suffix(".json").write_text('{"model_name": "test_lr", "metrics": {"test_roc_auc": 0.5}}')
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(path))
    load_model.cache_clear()
    from api.app import app
    with TestClient(app) as c:
        yield c
    mp.undo()
    load_model.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model"] == "test_lr"


def test_predict_ok(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert "x-request-id" in r.headers


def test_predict_allows_missing_optional_fields(client, sample_patient):
    p = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=p).status_code == 200


@pytest.mark.parametrize("field,value", [("cp", 9), ("age", -5), ("sex", 3), ("chol", "high")])
def test_predict_rejects_bad_input(client, sample_patient, field, value):
    assert client.post("/predict", json={**sample_patient, field: value}).status_code == 422


def test_predict_missing_required_field(client, sample_patient):
    p = dict(sample_patient)
    p.pop("age")
    assert client.post("/predict", json=p).status_code == 422


def test_batch(client, sample_patient):
    r = client.post("/predict/batch", json=[sample_patient] * 3)
    assert r.status_code == 200 and len(r.json()) == 3


def test_metrics_exposed(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
