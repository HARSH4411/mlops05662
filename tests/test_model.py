import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.data import clean, split_xy
from heart.features import build_preprocessor
from heart.predict import predict_records
from heart.train import candidate_models, make_pipeline


def test_preprocessor_output_is_finite(raw_df):
    X, _ = split_xy(clean(raw_df))
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert np.isfinite(Xt).all(), "imputation should remove all NaNs"


def test_preprocessor_handles_unseen_category(raw_df):
    X, _ = split_xy(clean(raw_df))
    pre = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["thal"] = 99
    assert pre.transform(unseen).shape[1] == pre.transform(X.head(1)).shape[1]


def test_candidate_models_has_at_least_two():
    assert len(candidate_models()) >= 2


@pytest.mark.parametrize("name", list(candidate_models().keys()))
def test_each_model_fits_and_predicts_proba(raw_df, name):
    X, y = split_xy(clean(raw_df))
    est, _ = candidate_models()[name]
    model = make_pipeline(est).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_predict_records_contract(raw_df, sample_patient):
    X, y = split_xy(clean(raw_df))
    model = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    out = predict_records([sample_patient, {**sample_patient, "ca": None, "thal": None}], model=model)
    assert len(out) == 2
    for r in out:
        assert r["prediction"] in (0, 1)
        assert 0 <= r["risk_probability"] <= 1
        assert 0.5 <= r["confidence"] <= 1
        assert r["label"] in ("disease", "no_disease")


def test_real_model_beats_baseline_if_present():
    """Quality gate: the trained artefact must clear a minimum ROC-AUC."""
    meta_p = config.MODEL_PATH.with_suffix(".json")
    if not meta_p.exists():
        pytest.skip("no trained model yet")
    import json
    meta = json.loads(meta_p.read_text())
    assert meta["metrics"]["test_roc_auc"] >= 0.80
