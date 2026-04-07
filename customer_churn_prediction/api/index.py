import os
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-random_forest")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

model = None
feature_cols = None

try:
    model = mlflow.sklearn.load_model(model_uri)

    # Retrieve feature columns logged during training from MLflow
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_version = next((v for v in versions if v.current_stage == MODEL_STAGE), None)
    if prod_version:
        run = client.get_run(prod_version.run_id)
        feature_cols_str = run.data.params.get("features", "")
        feature_cols = eval(feature_cols_str) if feature_cols_str else None
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
        model_stage=MODEL_STAGE,
    )
