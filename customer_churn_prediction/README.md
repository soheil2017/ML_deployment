# Customer Churn Prediction

End-to-end ML deployment: **MLflow** for lifecycle management + **Vercel** for production API.

## Architecture

```
Data → Preprocess → Train (MLflow tracking) → Model Registry → Vercel API (prod)
```

## Project Structure

```
customer_churn_prediction/
├── data/                    # Drop your churn.csv here (gitignored)
├── models/                  # Scaler + feature artifacts (gitignored)
├── notebooks/               # Exploration
├── src/
│   ├── preprocess.py        # Data preprocessing pipeline
│   ├── train.py             # Train models + log to MLflow
│   └── register_model.py    # Promote best model to Production
├── api/
│   └── index.py             # FastAPI app (Vercel serverless)
├── vercel.json              # Vercel deployment config
├── .env.example             # Environment variable template
└── requirements.txt
```

## Workflow

### 1. Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your paths and MLflow URI
```

### 2. Start MLflow Tracking Server (local)
```bash
mlflow server --host 0.0.0.0 --port 5000
# UI available at http://localhost:5000
```

### 3. Add your dataset
```
data/churn.csv   ← drop dataset here (must have a "churn" target column)
```

### 4. Train & Track Experiments
```bash
python src/train.py
# Trains RandomForest, GradientBoosting, LogisticRegression
# All runs tracked in MLflow with metrics, params, and artifacts
```

### 5. Register Best Model to Production
```bash
python src/register_model.py --run-id <RUN_ID> --model-name churn-random_forest
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
Staging → Production → Archived
```

Promote models using `src/register_model.py` or via the MLflow UI.
