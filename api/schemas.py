"""Request/response models. Ranges follow the UCI data dictionary (with slack)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (1 typical angina .. 4 asymptomatic)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-3, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[float] = Field(None, ge=0, le=3, description="Major vessels coloured by fluoroscopy (nullable)")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect (nullable)")

    model_config = {
        "json_schema_extra": {
            "example": {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
                        "restecg": 2, "thalach": 125, "exang": 1, "oldpeak": 2.4, "slope": 2,
                        "ca": 2, "thal": 7}
        }
    }


class Prediction(BaseModel):
    prediction: int
    label: str
    risk_probability: float
    confidence: float
    model_name: str
    model_version: str
    request_id: str
