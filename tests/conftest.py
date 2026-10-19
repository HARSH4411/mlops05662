import numpy as np
import pandas as pd
import pytest

from heart import config


@pytest.fixture(scope="session")
def raw_df():
    """Small synthetic frame in the raw UCI layout (incl. '?'-style NaNs and bad codes)."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(80, 200, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "num": rng.integers(0, 5, n),
    })
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    df.loc[2, "cp"] = 9  # invalid code
    return df[config.RAW_COLUMNS]


@pytest.fixture
def sample_patient():
    return {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
            "restecg": 2, "thalach": 125, "exang": 1, "oldpeak": 2.4, "slope": 2,
            "ca": 2, "thal": 7}
