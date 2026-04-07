import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, preprocess

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/churn.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn")

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
}


def train_and_log(model_name: str, model, X_train, X_test, y_train, y_test, feature_cols):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_param("features", feature_cols)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0,
        }
        mlflow.log_metrics(metrics)

        print(f"\n[{model_name}]")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"churn-{model_name}",
        )

        return metrics["roc_auc"], mlflow.active_run().info.run_id


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)

    # Save scaler as a shared artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    best_auc, best_run_id, best_model_name = 0, None, None
    for name, model in MODELS.items():
        auc, run_id = train_and_log(name, model, X_train, X_test, y_train, y_test, feature_cols)
        if auc > best_auc:
            best_auc, best_run_id, best_model_name = auc, run_id, name

    print(f"\nBest model: {best_model_name} (ROC-AUC={best_auc:.4f}, run_id={best_run_id})")
    print("To promote to production, run: python src/register_model.py")


if __name__ == "__main__":
    main()
