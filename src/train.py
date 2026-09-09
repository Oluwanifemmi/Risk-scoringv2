import logging
import os
import warnings
from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from xgboost import XGBClassifier

from src.preprocessing import data_import, data_split
from src.Feature_engineering import (
    build_preprocessing_pipeline,
    get_low_iv_columns,
    drop_columns,
    save_processed_data,
)
from src.evaluate import gini_coefficient, plot_ks

logger = logging.getLogger(__name__)

DATA_PATH = "data/application_train.csv"
MODEL_OUTPUT_DIR = "model"
MODEL_OUTPUT_PATH = f"{MODEL_OUTPUT_DIR}/riskscore.pkl"


def build_pipeline(scale_pos_weight: float) -> imbpipeline:
    """Assemble the full modeling pipeline: preprocessing, SMOTE, scaling, XGBoost."""
    preprocess = build_preprocessing_pipeline()

    pipe = imbpipeline(steps=[
        ('ColumnTransformer', preprocess),
        ('SMOTE', SMOTE(sampling_strategy='minority', random_state=42)),
        ('SCALER', StandardScaler()),
        ('XGBOOST', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eta=0.1,
            max_depth=10,
            subsample=1.0,
            min_child_weight=5,
            colsample_bytree=0.8,
            objective='binary:logistic',
        )),
    ])
    return pipe


def tune_pipeline(X_train, y_train, pipe: imbpipeline) -> Tuple[imbpipeline, Dict, float]:
    """Randomized hyperparameter search over the XGBOOST step.

    Returns the best pipeline, its best params, and its best CV score,
    so the caller can log all three to MLflow.
    """
    grid: Dict[str, List] = {
        'XGBOOST__learning_rate': [0.01, 0.05, 0.1, 0.3],
        'XGBOOST__n_estimators': [100, 300, 500],
        'XGBOOST__max_depth': [4, 6, 8, 10],
        'XGBOOST__subsample': [0.6, 0.7, 0.8, 1.0],
        'XGBOOST__colsample_bytree': [0.6, 0.8, 1.0],
        'XGBOOST__min_child_weight': [1, 3, 5],
    }

    warnings.filterwarnings('ignore')
    X_sample, y_sample = resample(X_train, y_train, n_samples=1000, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=20,
        cv=5,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_sample, y_sample)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")

    best_pipe = search.best_estimator_
    best_pipe.set_output(transform="pandas")
    return best_pipe, search.best_params_, search.best_score_


def fit_and_save(X_train, y_train, pipe: imbpipeline, output_path: str = MODEL_OUTPUT_PATH) -> imbpipeline:
    """Fit the pipeline on training data and persist it to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, output_path)
    logger.info(f"Saved fitted pipeline to {output_path}")
    return pipe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mlflow.set_experiment("credit_risk_model")

    with mlflow.start_run():
        # 1. Load raw data
        data = data_import(DATA_PATH)

        # 2. Split
        X_train, X_test, y_train, y_test = data_split(data)

        # 3. Decide which columns to drop, using ONLY training data
        columns_to_drop = get_low_iv_columns(X_train, y_train)
        mlflow.log_param("num_columns_dropped", len(columns_to_drop))

        # 4. Apply the SAME drop list to both train and test
        X_train = drop_columns(X_train, columns_to_drop)
        X_test = drop_columns(X_test, columns_to_drop)

        # 4b. Save the processed data for evaluate.py to reuse
        save_processed_data(X_train, X_test, y_train, y_test)

        # 5. Build the pipeline
        scale_pos_weight = 1.0  # placeholder — revisit this value
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        pipe = build_pipeline(scale_pos_weight=scale_pos_weight)

        # 6. Tune, and log what tuning found
        pipe, best_params, best_cv_score = tune_pipeline(X_train, y_train, pipe)
        mlflow.log_params(best_params)
        mlflow.log_metric("tuning_cv_auc", best_cv_score)

        # 7. Fit and save (joblib, as before)
        pipe = fit_and_save(X_train, y_train, pipe)

        # 8. Also log the fitted pipeline to MLflow, alongside the joblib file
        mlflow.sklearn.log_model(pipe, "model", serialization_format="pickle")

        # 9. Evaluate on the held-out test set and log those metrics
        ks_stat, p_value = plot_ks(X_test, y_test, pipe)
        gini = gini_coefficient(X_test, y_test, pipe)

        mlflow.log_metric("ks_statistic", ks_stat)
        mlflow.log_metric("gini_coefficient", gini)

        logger.info("MLflow run complete.")

        