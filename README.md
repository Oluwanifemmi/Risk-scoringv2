# Risk Scoring v2 : Customer Default Risk Model

A machine learning service that ranks customers by their likelihood of default, built for use in credit risk decisioning. The model is trained with XGBoost, tracked with MLflow, versioned with DVC, and served as a REST API with FastAPI.

## Overview

This project trains a binary classification model that scores customers on their probability of default and ranks them by risk level. It is designed to plug into a credit decisioning pipeline, where the output score can be used to approve, decline, or flag applications for manual review.

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing & features | pandas, numpy, feature-engine |
| Class imbalance handling | imbalanced-learn |
| Modeling | scikit-learn, XGBoost |
| Experiment tracking & model registry | MLflow |
| Data versioning | DVC |
| Serving | FastAPI, Uvicorn, Pydantic |
| Packaging | Docker |
| ML Pipeline | Airflow |
| Code Validation | Git Actions |

See `requirements.txt` for pinned versions.

## Project Structure

```
.
├── data/                  # Raw/processed data (tracked via DVC, not committed to git)
├── data.dvc               # DVC pointer file for the data directory
├── mlflow.db              # Local MLflow tracking store (experiments, runs, metrics, model registry)
├── src/
│   └── train.py           # Training pipeline (feature engineering, imbalance handling, XGBoost training)
├── main.py                # FastAPI application entrypoint (model inference API)
├── Dockerfile             # Container definition for serving the model via FastAPI/Uvicorn
├── requirements.txt        # Python dependencies
├── .dvcignore
├── .gitignore
└── README.md
```

> Note: `data/` is excluded from git (see `.gitignore`) and tracked separately with DVC.

## Model Details

- **Algorithm:** XGBoost classifier
- **Class imbalance:** handled via `imbalanced-learn`
- **Feature engineering:** built with `feature-engine`
- **Experiment tracking:** MLflow, experiment name `credit_risk_model`
- **Model registry:** registered under the name `credit-risk-model` in the MLflow Model Registry
- **Evaluation metrics:** KS statistic, Gini coefficient, and cross-validated AUC during hyperparameter tuning

Example hyperparameters explored across tuning runs (`config_1`–`config_4`):

| Parameter | Example value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 10 |
| `learning_rate` | 0.05 |
| `subsample` | 0.6 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `scale_pos_weight` | 1.0 |

Best-performing configuration so far reached a Gini coefficient of ~0.49 and a KS statistic of ~0.36 on the evaluation set.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Oluwanifemmi/Risk-scoringv2.git
cd Risk-scoringv2
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Pull the data (DVC)

Data is version-controlled with DVC and not stored in git. Make sure your DVC remote is configured, then:

```bash
dvc pull
```

### 4. Train the model

```bash
python src/train.py
```

Training runs, parameters, and metrics are logged automatically to MLflow (`mlflow.db`). To explore runs in the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://localhost:5000](http://localhost:5000).

## Serving the Model

The trained model is served through a FastAPI application (`main.py`).

### Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t risk-scoring-api .
docker run -p 8000:8000 risk-scoring-api
```

Once running, interactive API docs are available at:

```
http://localhost:8000/docs
```

## Data Versioning (DVC)

The `data.dvc` file tracks the `data/` directory (5 files, ~233 MB). Run `dvc pull` after cloning to fetch the dataset from the configured remote, and `dvc push` after updating local data to publish changes.
