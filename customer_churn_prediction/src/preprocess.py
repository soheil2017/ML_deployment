import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(db_path: str, table: str = "churn") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def preprocess(df: pd.DataFrame, target_col: str = "churn"):
    df = df.copy()

    # Drop ID-like columns (single unique value per row is useless as a feature)
    id_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    if id_cols:
        print(f"Dropping ID columns: {id_cols}")
        df = df.drop(columns=id_cols)

    # Drop rows with missing values
    df = df.dropna()

    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle string binary targets like "Yes"/"No"
    if y.dtype == object:
        unique_vals = y.unique()
        if set(unique_vals) <= {"Yes", "No"}:
            y = y.map({"Yes": 1, "No": 0})
        else:
            raise ValueError(f"Cannot auto-encode target with values: {unique_vals}")

    y = y.astype(int)

    # Encode categoricals
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)
