"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeartFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression vs rest")
    slope: int = Field(..., ge=0, le=3, description="ST slope")
    ca: int = Field(..., ge=0, le=4, description="Major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=7, description="Thalassemia indicator")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str
    probability_disease: float
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    version: str
