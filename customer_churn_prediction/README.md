# Customer Churn Prediction

End-to-end ML deployment: **MLflow** for lifecycle management + **Vercel** for production API.

## Architecture

```
Data → Preprocess → Train (MLflow tracking) → [Staging] → Evaluate (threshold gate) → [Production] → Vercel API
```

## Project Structure

```
customer_churn_prediction/
├── data/                    # Drop your churn.csv here (gitignored)
├── notebooks/               # Exploration
├── src/
│   ├── preprocess.py        # Data preprocessing pipeline
│   ├── train.py             # Train models + log to MLflow
│   ├── evaluate.py          # Evaluate + promote to Production if thresholds pass
│   └── register_model.py    # Manual promotion by run ID
├── api/
│   └── index.py             # FastAPI app (Vercel serverless)
├── vercel.json              # Vercel deployment config
├── .env                     # Environment variables (gitignored)
└── requirements.txt
```

## Workflow

### 1. Setup
```bash
pip install -r requirements.txt
# Edit .env with your paths and MLflow URI
```

### 2. Start MLflow Tracking Server (local)
```bash
mlflow server --host 0.0.0.0 --port 5000
# UI available at http://localhost:5000
```

### 3. Add your dataset
```
data/churn.csv   ← drop dataset here
```
Set the target column name in `.env`:
```
TARGET_COL=churn   # change to match your CSV column name
```

### 4. Train & Track Experiments
```bash
python src/train.py
# Trains RandomForest, GradientBoosting, LogisticRegression
# All runs tracked in MLflow with metrics, params, and artifacts
```

### 5. Evaluate & Promote to Production
```bash
# Evaluate a Staging model — exits with error if thresholds not met
python src/evaluate.py --model-name churn-random_forest --stage Staging

# Evaluate and auto-promote to Production if thresholds pass
python src/evaluate.py --model-name churn-random_forest --stage Staging --promote

# Or evaluate a specific run by ID
python src/evaluate.py --model-name churn-random_forest --run-id <RUN_ID> --promote
```

Thresholds are configurable in `.env`:
```
THRESHOLD_ACCURACY=0.80
THRESHOLD_F1=0.75
THRESHOLD_ROC_AUC=0.80
```

### 6. Run API Locally
```bash
uvicorn api.index:app --reload
# POST http://localhost:8000/predict
```

### 7. Deploy to Vercel
```bash
vercel login
vercel env add MLFLOW_TRACKING_URI   # your hosted MLflow URI
vercel env add MLFLOW_MODEL_NAME
vercel --prod
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status + model info |
| GET | `/health` | Health check |
| POST | `/predict` | Churn prediction |

### Predict Request
```json
{
  "features": {
    "tenure": 12,
    "monthly_charges": 65.5,
    "total_charges": 786.0
  }
}
```

### Predict Response
```json
{
  "churn": false,
  "probability": 0.1823,
  "model_name": "churn-random_forest",
  "model_stage": "Production"
}
```

## MLflow Registry Stages

```
[Run] → Staging → Production → Archived
```

Use `src/evaluate.py --promote` to promote through stages with automatic threshold validation,
or promote manually via the MLflow UI at `http://localhost:5000`.
