import json
import os
import pytest
from fastapi.testclient import TestClient

# Bundle directory
BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "../bundle")


# --- Bundle validation ---

def test_bundle_model_exists():
    assert os.path.isfile(os.path.join(BUNDLE_DIR, "model.pkl")), \
        "bundle/model.pkl not found — run: python3 src/export_model.py"


def test_bundle_feature_cols_exists():
    assert os.path.isfile(os.path.join(BUNDLE_DIR, "feature_cols.json")), \
        "bundle/feature_cols.json not found — run: python3 src/export_model.py"


def test_bundle_meta_exists():
    assert os.path.isfile(os.path.join(BUNDLE_DIR, "meta.json")), \
        "bundle/meta.json not found — run: python3 src/export_model.py"


def test_bundle_meta_has_required_keys():
    with open(os.path.join(BUNDLE_DIR, "meta.json")) as f:
        meta = json.load(f)
    for key in ["model_name", "model_alias", "model_version", "run_id"]:
        assert key in meta, f"meta.json missing key: {key}"


def test_bundle_feature_cols_is_list():
    with open(os.path.join(BUNDLE_DIR, "feature_cols.json")) as f:
        cols = json.load(f)
    assert isinstance(cols, list)
    assert len(cols) > 0


# --- API endpoint tests ---

@pytest.fixture(scope="module")
def client():
    from api.index import app
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json={
        "features": {
            "tenure": 12,
            "monthlycharges": 65.5,
            "totalcharges": 786.0,
            "seniorcitizen": 0
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert "churn" in data
    assert "probability" in data
    assert isinstance(data["churn"], bool)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_features_defaults_to_zero(client):
    # Sending empty features should not crash — missing cols default to 0
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 200


def test_predict_high_risk_customer(client):
    # Short tenure + high charges + month-to-month contract = likely churner
    response = client.post("/predict", json={
        "features": {
            "tenure": 1,
            "monthlycharges": 100.0,
            "totalcharges": 100.0,
            "contract_Month-to-month": 1,
            "seniorcitizen": 0
        }
    })
    assert response.status_code == 200
    assert response.json()["probability"] > 0.1
