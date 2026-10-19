"""Model training with hyper-parameter search, cross-validation and MLflow tracking.

Usage:  python -m heart.train [--experiment heart-disease] [--no-mlflow]

For every candidate model we:
  1. GridSearchCV (5-fold stratified, scoring = ROC-AUC) on the training split
  2. Log params, CV metrics, held-out test metrics, ROC / confusion-matrix plots and
     the fitted pipeline to MLflow (one run per model)
The model with the best mean CV ROC-AUC is saved to models/model.joblib together
with a metadata JSON, and registered in the MLflow model registry.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean, split_xy  # noqa: E402
from heart.features import build_preprocessor  # noqa: E402

log = logging.getLogger("heart.train")
warnings.filterwarnings("ignore", message=".*Integer columns.*")
warnings.filterwarnings("ignore", category=FutureWarning)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def candidate_models(seed: int = config.SEED) -> dict:
    """Model zoo + search grids. Recall is weighted heavily in the write-up
    because a missed disease case (false negative) is the costly error."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
            {"clf__C": [0.01, 0.1, 0.5, 1.0, 5.0], "clf__penalty": ["l1", "l2"],
             "clf__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 4, 8],
             "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.5]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
             "clf__max_depth": [2, 3], "clf__subsample": [0.8, 1.0]},
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("clf", estimator)])


def test_metrics(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def plot_diagnostics(name, model, X, y, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title(f"ROC curve - {name}")
    p = outdir / f"roc_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    p = outdir / f"cm_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    clf = model.named_steps["clf"]
    names = model.named_steps["pre"].get_feature_names_out()
    imp = getattr(clf, "feature_importances_", None)
    if imp is None and hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    if imp is not None:
        order = np.argsort(imp)[-15:]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([names[i] for i in order], imp[order], color="#66c2a5")
        ax.set_title(f"Top features - {name}")
        p = outdir / f"importance_{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def train(experiment: str = "heart-disease", use_mlflow: bool = True, quick: bool = False,
          model_path=config.MODEL_PATH) -> dict:
    df = load_clean()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    if use_mlflow:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(experiment)

    results = {}
    for name, (est, grid) in candidate_models().items():
        if quick:  # tiny grid for CI / unit tests
            grid = {k: v[:1] for k, v in grid.items()}
        t0 = time.time()
        search = GridSearchCV(make_pipeline(est), grid, cv=cv, scoring=SCORING,
                              refit="roc_auc", n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        i = search.best_index_
        cvm = {f"cv_{m}_mean": float(search.cv_results_[f"mean_test_{m}"][i]) for m in SCORING}
        cvm.update({f"cv_{m}_std": float(search.cv_results_[f"std_test_{m}"][i]) for m in SCORING})
        tm = test_metrics(best, X_te, y_te)
        metrics = {**cvm, **tm, "train_seconds": time.time() - t0}
        results[name] = {"model": best, "params": search.best_params_, "metrics": metrics}
        log.info("%-20s cv_auc=%.3f+/-%.3f test_auc=%.3f test_recall=%.3f params=%s",
                 name, cvm["cv_roc_auc_mean"], cvm["cv_roc_auc_std"], tm["test_roc_auc"],
                 tm["test_recall"], search.best_params_)

        if use_mlflow:
            with mlflow.start_run(run_name=name) as run:
                mlflow.set_tags({"model_family": name, "grid_size": len(search.cv_results_["params"])})
                mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
                mlflow.log_params({"test_size": 0.2, "cv_folds": 5, "seed": config.SEED,
                                   "n_train": len(X_tr), "n_test": len(X_te)})
                mlflow.log_metrics(metrics)
                for p in plot_diagnostics(name, best, X_te, y_te, config.FIGURES_DIR):
                    mlflow.log_artifact(str(p), artifact_path="plots")
                # cloudpickle: we only ever load models this pipeline produced itself
                mlflow.sklearn.log_model(best, name="model",
                                         input_example=X_te.head(3).astype(float),
                                         serialization_format="cloudpickle")
                results[name]["run_id"] = run.info.run_id
        else:
            plot_diagnostics(name, best, X_te, y_te, config.FIGURES_DIR)

    best_name = max(results, key=lambda n: results[n]["metrics"]["cv_roc_auc_mean"])
    best = results[best_name]
    model_path = config.MODEL_PATH if model_path is None else model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_path)
    meta = {
        "model_name": best_name,
        "params": best["params"],
        "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
        "features": config.FEATURES,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": best.get("run_id"),
    }
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))

    comparison = {n: {k: round(v, 4) for k, v in r["metrics"].items()} for n, r in results.items()}
    (config.ROOT / "reports").mkdir(exist_ok=True)
    (config.ROOT / "reports" / "model_comparison.json").write_text(json.dumps(comparison, indent=2))

    if use_mlflow and best.get("run_id"):
        try:
            mlflow.register_model(f"runs:/{best['run_id']}/model", "heart-disease-classifier")
        except Exception as exc:  # registry is optional (e.g. file store)
            log.warning("Model registry skipped: %s", exc)

    log.info("BEST MODEL: %s  -> saved to %s", best_name, model_path)
    return meta


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="heart-disease")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--quick", action="store_true", help="1-point grids (CI smoke run)")
    a = p.parse_args()
    train(a.experiment, use_mlflow=not a.no_mlflow, quick=a.quick)


if __name__ == "__main__":
    main()
