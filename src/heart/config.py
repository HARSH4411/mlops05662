"""Central configuration: paths, column schema and random seed."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_PATH = DATA_DIR / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
FIGURES_DIR = ROOT / "reports" / "figures"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Continuous measurements -> imputed with median and standard-scaled
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Already 0/1 -> passed through after imputation
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes -> one-hot encoded
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

SEED = int(os.getenv("SEED", "42"))
