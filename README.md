# ML Deployment Projects

A collection of end-to-end machine learning, deep learning, and NLP projects — each deployed to production with a live API.

The focus of this repository is not just building models, but demonstrating the **full production pipeline** that real ML teams follow:

```
Data Ingestion → Preprocessing → Experimentation → Evaluation → Registry → Deployment → Live API
```

Every project in this repository uses:
- **MLflow** — experiment tracking, model versioning, and alias-based promotion (`@challenger` → `@champion`)
- **Vercel** — serverless production API deployment
- **Threshold-gated promotion** — models must pass automated metric checks before reaching production

---

## Projects

### 1. Customer Churn Prediction
> Binary classification — predicts whether a telecom customer will cancel their subscription.

| | |
|---|---|
| **Dataset** | IBM Telco Customer Churn (7,043 customers, 20 features) |
| **Model** | Gradient Boosting Classifier |
| **Performance** | Accuracy: 0.80 · F1: 0.57 · ROC-AUC: 0.85 |
| **Stack** | SQLite · scikit-learn · MLflow · FastAPI · Vercel |
| **Live API** | https://customer-churn-prediction-omega-opal.vercel.app |
| **Swagger UI** | https://customer-churn-prediction-omega-opal.vercel.app/docs |

📁 [`customer_churn_prediction/`](./customer_churn_prediction)

---

## Repository Structure

```
Deployment/
└── customer_churn_prediction/    # Project 1 — binary classification
    ├── data/                     # Raw dataset (gitignored)
    ├── bundle/                   # Exported model artifacts for Vercel
    ├── src/                      # Training, evaluation, and export scripts
    ├── api/                      # FastAPI serverless endpoint
    └── README.md                 # Full pipeline documentation
```

New projects will be added as top-level folders following the same structure.

---

## Shared Pipeline Pattern

Each project follows the same production workflow:

```
1. Data Ingestion     →  Load raw data into SQLite
2. Preprocessing      →  Clean, encode, scale
3. Training           →  Train multiple models, log all runs to MLflow
4. Evaluation         →  Automated metric thresholds gate promotion
5. Registry           →  Best model assigned @champion alias in MLflow
6. Export             →  Champion model exported to bundle/
7. Deployment         →  Vercel serves the bundled model via FastAPI
```

---

## Tech Stack

| Tool | Role |
|---|---|
| **Python** | Core language |
| **scikit-learn** | ML models |
| **MLflow** | Experiment tracking · Model registry · Alias promotion |
| **FastAPI** | REST API |
| **SQLite** | Lightweight data storage |
| **Vercel** | Serverless production deployment |
| **GitHub** | Version control + CI/CD trigger |
