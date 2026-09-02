import logging
import warnings
from typing import Dict, List
from Feature_engineering import build_preprocessing_pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
import joblib

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = "riskscore.pkl"

#combining the pipeline for the algorithm
def build_pipeline(scale_pos_weight: float):
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

#hyperparameter tunning


def tune_pipeline(X_train, y_train, pipe: imbpipeline):
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
    X_sample, y_sample = resample(X_train, y_train, n_samples=1000, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=20,
        cv=5,
        n_jobs=-1,
        random_state=42,)
    
    search.fit(X_sample, y_sample)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")

    best_pipe = search.best_estimator_
    best_pipe.set_output(transform="pandas")
    return best_pipe      
  

#fitting the model into the algorithm
def fit_and_save(X_train, y_train, pipe: imbpipeline, output_path: str = MODEL_OUTPUT_PATH) -> imbpipeline:
    """Fit the pipeline on training data and persist it to disk."""
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, output_path)
    logger.info(f"Saved fitted pipeline to {output_path}")
    return pipe