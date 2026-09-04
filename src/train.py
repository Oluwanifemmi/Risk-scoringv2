import logging
import warnings
from typing import Dict, List
import joblib
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
    drop_columns,save_processed_data
)
import os

logger = logging.getLogger(__name__)

DATA_PATH = "data/application_train.csv"
MODEL_OUTPUT_PATH = "riskscore.pkl"


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


def tune_pipeline(X_train, y_train, pipe: imbpipeline) -> imbpipeline:
    """Randomized hyperparameter search over the XGBOOST step."""
    grid: Dict[str, List] = {
        'XGBOOST__learning_rate': [0.01, 0.05, 0.1, 0.3],
        'XGBOOST__n_estimators': [100, 300, 500],
        'XGBOOST__max_depth': [4, 6, 8, 10],
        'XGBOOST__subsample': [0.6, 0.7, 0.8, 1.0],
        'XGBOOST__colsample_bytree': [0.6, 0.8, 1.0],
        'XGBOOST__min_child_weight': [1, 3, 5],
    }

    warnings.filterwarnings('ignore')
    X_sample, y_sample = resample(X_train, y_train, n_samples=100, random_state=42)

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
    return best_pipe


def fit_and_save(X_train, y_train, pipe: imbpipeline, output_path: str = MODEL_OUTPUT_PATH):
    """Fit the pipeline on training data and persist it to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, output_path)
    logger.info(f"Saved fitted pipeline to {output_path}")
    return pipe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. Load raw data
    data = data_import(DATA_PATH)

    # 2. Split
    X_train, X_test, y_train, y_test = data_split(data)

    # 3. Decide which columns to drop, using ONLY training data
    columns_to_drop = get_low_iv_columns(X_train, y_train)

    # 4. Apply the SAME drop list to both train and test
    X_train = drop_columns(X_train, columns_to_drop)
    X_test = drop_columns(X_test, columns_to_drop)

    #4b saving the processed data for evaluation
    save_processed_data(X_train, X_test, y_train, y_test)

    # 5. Build the pipeline
    pipe = build_pipeline(scale_pos_weight=1.0)  #placeholder, see note below

    # 6. (Optional) tune — comment out if you want to skip this for a quick run
    pipe = tune_pipeline(X_train, y_train, pipe)

    # 7. Fit and save
    pipe = fit_and_save(X_train, y_train, pipe)