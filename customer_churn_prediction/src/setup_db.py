"""
One-time script to load the churn CSV into a SQLite database.

Usage:
    python src/setup_db.py --csv data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python src/setup_db.py --csv data/WA_Fn-UseC_-Telco-Customer-Churn.csv --db data/churn.db
"""
import argparse
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_db(csv_path: str, db_path: str, table: str = "churn"):
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Normalise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

    print(f"SQLite DB created: {db_path}")
    print(f"Table '{table}' loaded with {len(df)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the source CSV file")
    parser.add_argument(
        "--db",
        default=os.getenv("DB_PATH", os.path.join(ROOT, "data/churn.db")),
        help="Path for the SQLite DB (default: data/churn.db)",
    )
    parser.add_argument("--table", default="churn", help="Table name (default: churn)")
    args = parser.parse_args()

    create_db(args.csv, args.db, args.table)
