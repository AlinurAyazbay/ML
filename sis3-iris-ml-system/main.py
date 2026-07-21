"""
FastAPI service for Iris species prediction.

Loads the trained model from model.joblib on startup and exposes:
  - GET  /        : health check
  - GET  /model   : model metadata
  - POST /predict : classify an iris flower
"""

import os
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print(f"[startup] Loaded model from: {MODEL_PATH}")
except Exception as exc:  # pragma: no cover
    model = None
    MODEL_LOADED = False
    print(f"[startup] FAILED to load model: {exc}")

SPECIES_NAMES: Dict[int, str] = {0: "setosa", 1: "versicolor", 2: "virginica"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Iris Species Prediction API",
    description="Predict iris flower species from sepal and petal measurements.",
    version="2.0.0",
)

# Allow Streamlit frontend (running on a different port) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=15, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=15, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=15, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=15, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    }


class PredictionResponse(BaseModel):
    species: str
    species_id: int
    confidence: float
    probabilities: Dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "ML API is running",
        "model": "Logistic Regression",
        "dataset": "Iris",
        "status": "healthy" if MODEL_LOADED else "model_not_loaded",
    }


@app.get("/model")
def model_info():
    """Return metadata about the loaded model."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "classes": [SPECIES_NAMES[int(c)] for c in model.classes_],
        "n_features": int(model.n_features_in_),
        "feature_names": [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """Predict iris species from flower measurements."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(probabilities[prediction])

    return PredictionResponse(
        species=SPECIES_NAMES[prediction],
        species_id=prediction,
        confidence=confidence,
        probabilities={
            SPECIES_NAMES[i]: float(p) for i, p in enumerate(probabilities)
        },
    )
