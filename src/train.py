import logging
import os
import warnings
from typing import Dict, List

import joblib
import mlflow
import mlflow.sklearn
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
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
TOP_N_CONFIGS = 4


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


def get_top_n_configs(X_train, y_train, pipe: imbpipeline, n: int = TOP_N_CONFIGS) -> List[Dict]:
    """Run RandomizedSearchCV and return the top n param combinations by CV score,
    instead of only the single best one.
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

    # Pair every trial's params with its mean CV score, then sort descending.
    all_params = search.cv_results_['params']
    all_scores = search.cv_results_['mean_test_score']
    ranked = sorted(zip(all_params, all_scores), key=lambda pair: pair[1], reverse=True)

    top_configs = [params for params, score in ranked[:n]]

    logger.info(f"Top {n} CV scores: {[round(score, 4) for _, score in ranked[:n]]}")
    return top_configs


def fit_config(X_train, y_train, base_pipe: imbpipeline, config: Dict) -> imbpipeline:
    """Fit a fresh, independent clone of base_pipe using one specific param config."""
    pipe = clone(base_pipe)
    pipe.set_params(**config)
    pipe.fit(X_train, y_train)
    return pipe


def fit_and_save(pipe: imbpipeline, output_path: str = MODEL_OUTPUT_PATH) -> imbpipeline:
    """Persist an already-fitted pipeline to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipe, output_path)
    logger.info(f"Saved fitted pipeline to {output_path}")
    return pipe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mlflow.set_experiment("credit_risk_model")

    # 1. Load raw data
    data = data_import(DATA_PATH)

    # 2. Split
    X_train, X_test, y_train, y_test = data_split(data)

    # 3. Decide which columns to drop, using ONLY training data
    columns_to_drop = get_low_iv_columns(X_train, y_train)

    # 4. Apply the SAME drop list to both train and test
    X_train = drop_columns(X_train, columns_to_drop)
    X_test = drop_columns(X_test, columns_to_drop)

    # 4b. Save the processed data for evaluate.py to reuse
    save_processed_data(X_train, X_test, y_train, y_test)

    # 5. Build the base (untrained) pipeline shape
    scale_pos_weight = 1.0  # placeholder — revisit this value
    base_pipe = build_pipeline(scale_pos_weight=scale_pos_weight)

    # 6. Get the top N hyperparameter configs from search, instead of just 1
    top_configs = get_top_n_configs(X_train, y_train, base_pipe, n=TOP_N_CONFIGS)

    # 7. Fit each config as its own model, log each as an independent MLflow run,
    #    and track which one performs best on the REAL held-out test set (not CV).
    best_pipe = None
    best_gini = float("-inf")

    for i, config in enumerate(top_configs, start=1):
        with mlflow.start_run(run_name=f"config_{i}"):
            logger.info(f"Fitting config {i}/{len(top_configs)}: {config}")

            pipe = fit_config(X_train, y_train, base_pipe, config)

            mlflow.log_param("num_columns_dropped", len(columns_to_drop))
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_params(config)

            ks_stat, p_value = plot_ks(X_test, y_test, pipe)
            gini = gini_coefficient(X_test, y_test, pipe)

            mlflow.log_metric("ks_statistic", ks_stat)
            mlflow.log_metric("gini_coefficient", gini)
            mlflow.sklearn.log_model(pipe, "model", serialization_format="pickle")

            logger.info(f"Config {i} -> Gini: {gini:.4f}, KS: {ks_stat:.4f}")

            if gini > best_gini:
                best_gini = gini
                best_pipe = pipe

    # 8. Save only the best-performing pipeline to disk
    logger.info(f"Best Gini across all configs: {best_gini:.4f}")
    fit_and_save(best_pipe)