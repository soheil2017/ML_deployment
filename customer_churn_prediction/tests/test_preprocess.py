import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from preprocess import preprocess


def make_df(**kwargs):
    """Create a minimal synthetic churn dataframe."""
    size = kwargs.get("size", 100)
    np.random.seed(42)
    return pd.DataFrame({
        "customerid": [f"ID-{i}" for i in range(size)],
        "tenure": np.random.randint(1, 72, size),
        "monthlycharges": np.random.uniform(20, 120, size),
        "totalcharges": np.random.uniform(100, 8000, size),
        "seniorcitizen": np.random.randint(0, 2, size),
        "churn": np.random.choice(["Yes", "No"], size),
    })


def test_drops_id_columns():
    df = make_df()
    _, _, _, _, _, feature_cols = preprocess(df, target_col="churn")
    assert "customerid" not in feature_cols


def test_yes_no_target_encoded():
    df = make_df()
    _, _, y_train, y_test, _, _ = preprocess(df, target_col="churn")
    all_labels = pd.concat([y_train, y_test])
    assert set(all_labels.unique()).issubset({0, 1})


def test_output_shapes():
    df = make_df(size=200)
    X_train, X_test, y_train, y_test, _, _ = preprocess(df, target_col="churn")
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]


def test_missing_target_raises():
    df = make_df()
    with pytest.raises(ValueError, match="Target column 'wrong_col' not found"):
        preprocess(df, target_col="wrong_col")


def test_scaler_applied():
    df = make_df(size=200)
    X_train, X_test, _, _, scaler, _ = preprocess(df, target_col="churn")
    # StandardScaler: train mean should be close to 0
    assert abs(X_train.mean()) < 1.0
