import json
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Bundle directory — model artifacts exported from MLflow and committed to the repo
BUNDLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bundle")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.1")

model = None
feature_cols = None
meta = {}

try:
    model = joblib.load(os.path.join(BUNDLE_DIR, "model.pkl"))

    with open(os.path.join(BUNDLE_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    with open(os.path.join(BUNDLE_DIR, "meta.json")) as f:
        meta = json.load(f)

    print(f"Loaded model: {meta.get('model_name')}@{meta.get('model_alias')} v{meta.get('model_version')}")
except Exception as e:
    print(f"Warning: Could not load model — {e}")


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    churn: bool
    probability: float
    model_name: str
    model_version: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": meta.get("model_name"),
        "alias": meta.get("model_alias"),
        "version": meta.get("model_version"),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if feature_cols is None:
        raise HTTPException(status_code=503, detail="Feature columns not available")

    # Align input with training feature columns (fill missing with 0)
    x = np.array([request.features.get(col, 0) for col in feature_cols]).reshape(1, -1)

    prediction = model.predict(x)[0]
    probability = model.predict_proba(x)[0][1]

    return PredictResponse(
        churn=bool(prediction),
        probability=round(float(probability), 4),
        model_name=meta.get("model_name", "unknown"),
        model_version=str(meta.get("model_version", "unknown")),
    )
