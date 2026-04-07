import json
import os
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-gradient_boosting")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = None
feature_cols = None

try:
    model = mlflow.sklearn.load_model(model_uri)

    # Retrieve feature columns from the JSON artifact logged during training
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    artifact_path = client.download_artifacts(mv.run_id, "feature_cols.json")
    with open(artifact_path) as f:
        feature_cols = json.load(f)
except Exception as e:
    print(f"Warning: Could not load model at startup — {e}")


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    churn: bool
    probability: float
    model_name: str
    model_stage: str


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}


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
        model_name=MODEL_NAME,
        model_stage=MODEL_ALIAS,
    )
