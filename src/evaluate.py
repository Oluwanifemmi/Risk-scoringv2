import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


MODEL_PATH = "model/riskscore.pkl"
PROCESSED_DATA_DIR = "data/preprocessed"


def load_processed_data(data_dir: str = PROCESSED_DATA_DIR):
    """Load the saved post-drop, pre-pipeline train/test data."""
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze("columns")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def plot_ks(X_test, y_test, model):
    """Compute the KS statistic between predicted scores for good vs. default cases."""
    y_proba = model.predict_proba(X_test)[:, 1]
    scores_good = y_proba[y_test == 0]
    scores_default = y_proba[y_test == 1]
    ks_stat, p_value = ks_2samp(scores_good, scores_default)
    logger.info(f"KS Statistic: {ks_stat:.4f}")
    logger.info(f"P-value: {p_value:.4f}")
    return ks_stat, p_value


def gini_coefficient(X_test, y_test, model) -> float:
    """Compute the Gini coefficient from ROC AUC."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    gini = 2 * roc_auc_score(y_test, y_pred_proba) - 1
    logger.info(f"Gini coefficient: {gini:.4f}")
    return gini


def cross_validation(X_train, y_train, model) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified 5-fold CV, reporting both AUC and Gini per fold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    gini_scores = 2 * auc_scores - 1
    logger.info(f"CV AUC scores: {auc_scores}")
    logger.info(f"CV Gini scores: {gini_scores}")
    return auc_scores, gini_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test = load_processed_data()

    ks_stat, p_value = plot_ks(X_test, y_test, model)
    gini = gini_coefficient(X_test, y_test, model)
    auc_scores, gini_scores = cross_validation(X_train, y_train, model)

