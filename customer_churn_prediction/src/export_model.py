"""
Export the champion model from MLflow Registry into the bundle/ directory
so it can be committed and bundled with the Vercel deployment.

Run this before every Vercel deployment when a new champion is promoted.

Usage:
    python src/export_model.py
    python src/export_model.py --model-name churn-gradient_boosting --alias champion
"""
import argparse
import json
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-gradient_boosting")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
BUNDLE_DIR = os.path.join(ROOT, "bundle")


def export(model_name: str, alias: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Fetching model: {model_name}@{alias}")
    mv = client.get_model_version_by_alias(model_name, alias)
    print(f"Found version {mv.version} (run_id: {mv.run_id})")

    os.makedirs(BUNDLE_DIR, exist_ok=True)

    # Download and save model
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    model_path = os.path.join(BUNDLE_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    # Download feature_cols artifact
    artifact_path = client.download_artifacts(mv.run_id, "feature_cols.json")
    with open(artifact_path) as f:
        feature_cols = json.load(f)

    feature_cols_path = os.path.join(BUNDLE_DIR, "feature_cols.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Feature cols saved: {feature_cols_path} ({len(feature_cols)} features)")

    # Download scaler artifact
    scaler_artifact = client.download_artifacts(mv.run_id, "scaler.pkl")
    import shutil
    shutil.copy(scaler_artifact, os.path.join(BUNDLE_DIR, "scaler.pkl"))
    print(f"Scaler saved: {os.path.join(BUNDLE_DIR, 'scaler.pkl')}")

    # Save bundle metadata
    meta = {
        "model_name": model_name,
        "model_alias": alias,
        "model_version": mv.version,
        "run_id": mv.run_id,
    }
    with open(os.path.join(BUNDLE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBundle ready in: {BUNDLE_DIR}")
    print("Next step: git add bundle/ && git commit && vercel --prod")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--alias", default=MODEL_ALIAS)
    args = parser.parse_args()
    export(args.model_name, args.alias)
