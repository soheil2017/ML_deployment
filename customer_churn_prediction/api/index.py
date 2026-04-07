import os
import joblib
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

# Load model and scaler once at startup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

try:
    model = mlflow.sklearn.load_model(model_uri)
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
except Exception as e:
    model, scaler, feature_cols = None, None, None
    print(f"Warning: Could not load model at startup — {e}")


class PredictRequest(BaseModel):
    features: dict  # {feature_name: value}


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

    # Align input features with training columns
    input_df = {col: request.features.get(col, 0) for col in feature_cols}
    x = np.array(list(input_df.values())).reshape(1, -1)
    x_scaled = scaler.transform(x)

    prediction = model.predict(x_scaled)[0]
    probability = model.predict_proba(x_scaled)[0][1]

    return PredictResponse(
        churn=bool(prediction),
        probability=round(float(probability), 4),
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )
