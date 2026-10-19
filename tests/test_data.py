import pandas as pd

from heart import config
from heart.data import clean, load_raw, split_xy


def test_clean_creates_binary_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert (out[config.TARGET] == (raw_df["num"] > 0).astype(int).values).all()


def test_clean_nulls_invalid_codes(raw_df):
    out = clean(raw_df)
    assert pd.isna(out.loc[2, "cp"])
    assert out["cp"].dropna().isin([1, 2, 3, 4]).all()


def test_clean_keeps_missing_for_pipeline(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() >= 1 and out["thal"].isna().sum() >= 1


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(5)])
    assert len(clean(doubled)) == len(clean(raw_df))


def test_split_xy_columns(raw_df):
    X, y = split_xy(clean(raw_df))
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y)


def test_load_raw_parses_question_marks(tmp_path):
    p = tmp_path / "raw.data"
    p.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = load_raw(p)
    assert df.shape == (1, 14) and pd.isna(df.loc[0, "ca"])
