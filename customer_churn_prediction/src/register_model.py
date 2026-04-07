"""
Promote the best run's model from MLflow Model Registry to 'Production' stage.
Usage:
    python src/register_model.py --run-id <RUN_ID> --model-name churn-random_forest
"""
import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def promote_to_production(run_id: str, model_name: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get latest version for this model
    versions = client.search_model_versions(f"name='{model_name}'")
    target = next((v for v in versions if v.run_id == run_id), None)

    if not target:
        raise ValueError(f"No version found for run_id={run_id} in model={model_name}")

    # Archive current production models
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(model_name, v.version, "Archived")

    # Promote target to Production
    client.transition_model_version_stage(model_name, target.version, "Production")
    print(f"Model '{model_name}' version {target.version} promoted to Production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    promote_to_production(args.run_id, args.model_name)
