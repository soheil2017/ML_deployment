"""
Evaluate a model from MLflow Model Registry against a dataset.
If all metrics pass the defined thresholds, the model is promoted to Production.

Usage:
    python src/evaluate.py --model-name churn-random_forest --stage Staging
    python src/evaluate.py --model-name churn-random_forest --run-id <RUN_ID>
"""
import argparse
import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, preprocess

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.getenv("DATA_PATH", "data/churn.csv")
TARGET_COL = os.getenv("TARGET_COL", "churn")

# Promotion thresholds — model must pass ALL to be promoted
THRESHOLDS = {
    "accuracy": float(os.getenv("THRESHOLD_ACCURACY", "0.80")),
    "f1_score": float(os.getenv("THRESHOLD_F1", "0.75")),
    "roc_auc": float(os.getenv("THRESHOLD_ROC_AUC", "0.80")),
}


def evaluate(model_name: str, stage: str = None, run_id: str = None, promote: bool = False):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Build model URI
    if run_id:
        model_uri = f"runs:/{run_id}/model"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        raise ValueError("Provide either --stage or --run-id")

    print(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Load scaler and feature columns saved during training
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

    # Load and preprocess evaluation data
    df = load_data(DATA_PATH)
    _, X_test, _, y_test, _, _ = preprocess(df, target_col=TARGET_COL)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0,
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
    }

    # Log evaluation run to MLflow
    with mlflow.start_run(run_name=f"evaluate-{model_name}"):
        mlflow.set_tag("evaluated_model", model_name)
        mlflow.set_tag("evaluated_stage", stage or "run")
        mlflow.log_metrics(metrics)

    # Print report
    print("\n" + "=" * 50)
    print(f"Evaluation Report — {model_name} [{stage or run_id}]")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    print(f"{'Metric':<12} {'Score':>8}   {'Threshold':>10}   {'Pass?':>6}")
    print("-" * 45)

    passed_all = True
    for metric, threshold in THRESHOLDS.items():
        score = metrics[metric]
        passed = score >= threshold
        passed_all = passed_all and passed
        status = "PASS" if passed else "FAIL"
        print(f"{metric:<12} {score:>8.4f}   {threshold:>10.4f}   {status:>6}")

    print("=" * 50)

    if passed_all:
        print("\nAll thresholds passed.")
        if promote:
            _promote_to_production(client, model_name, stage, run_id)
        else:
            print("Run with --promote to push this model to Production.")
    else:
        print("\nThreshold check FAILED. Model will NOT be promoted.")
        sys.exit(1)


def _promote_to_production(client: MlflowClient, model_name: str, stage: str, run_id: str):
    versions = client.search_model_versions(f"name='{model_name}'")

    if run_id:
        target = next((v for v in versions if v.run_id == run_id), None)
    else:
        target = next((v for v in versions if v.current_stage == stage), None)

    if not target:
        print("Could not find model version to promote.")
        sys.exit(1)

    # Archive existing Production models
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(model_name, v.version, "Archived")
            print(f"Archived version {v.version}")

    client.transition_model_version_stage(model_name, target.version, "Production")
    print(f"Model '{model_name}' version {target.version} promoted to Production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Registered model name in MLflow")
    parser.add_argument("--stage", default="Staging", help="Model stage to evaluate (default: Staging)")
    parser.add_argument("--run-id", default=None, help="Specific MLflow run ID to evaluate")
    parser.add_argument("--promote", action="store_true", help="Promote to Production if thresholds pass")
    args = parser.parse_args()

    evaluate(
        model_name=args.model_name,
        stage=args.stage if not args.run_id else None,
        run_id=args.run_id,
        promote=args.promote,
    )
