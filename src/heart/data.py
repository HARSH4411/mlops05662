"""Data acquisition and cleaning."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heart import config

log = logging.getLogger(__name__)

# Valid code sets from the UCI data dictionary
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}


def download(url: str = config.UCI_URL, dest: Path = config.RAW_PATH, force: bool = False) -> Path:
    """Download the raw Cleveland file from UCI (skips if it already exists)."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted public URL)
    return dest


def load_raw(path: Path = config.RAW_PATH) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned frame with a binary `target` column.

    - drops exact duplicate rows
    - casts all columns to numeric
    - replaces out-of-dictionary category codes with NaN (imputed later in the pipeline)
    - converts `num` (0-4 severity) into binary presence/absence
    Missing values are *not* imputed here so that imputation is learned
    inside the sklearn pipeline (no train/test leakage).
    """
    out = df.copy().drop_duplicates().reset_index(drop=True)
    out = out.apply(pd.to_numeric, errors="coerce")
    for col, codes in VALID_CODES.items():
        out.loc[~out[col].isin(codes) & out[col].notna(), col] = np.nan
    out.loc[(out["ca"] < 0) | (out["ca"] > 3), "ca"] = np.nan
    out[config.TARGET] = (out["num"] > 0).astype(int)
    return out.drop(columns=["num"])


def load_clean(path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET]
