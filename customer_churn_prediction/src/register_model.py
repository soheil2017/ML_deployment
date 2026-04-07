"""
Manually assign the 'champion' alias to a specific model version.
Use this for manual promotion without the evaluation threshold gate.

Usage:
    python src/register_model.py --run-id <RUN_ID> --model-name churn-gradient_boosting
    python src/register_model.py --run-id <RUN_ID> --model-name churn-gradient_boosting --alias champion
"""
import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def assign_alias(run_id: str, model_name: str, alias: str = "champion"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    target = next((v for v in versions if v.run_id == run_id), None)

    if not target:
        raise ValueError(f"No version found for run_id={run_id} in model={model_name}")

    client.set_registered_model_alias(model_name, alias, target.version)
    print(f"Model '{model_name}' version {target.version} assigned alias '{alias}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--alias", default="champion", help="Alias to assign (default: champion)")
    args = parser.parse_args()
    assign_alias(args.run_id, args.model_name, args.alias)
