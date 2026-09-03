import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
from src.preprocessing import data_split
import logging
from typing import Tuple

#loaded the save model
model = joblib.load('model/riskscore.pkl')

logger = logging.getLogger(__name__)
#ks to evaluate the model
def plot_ks(X_test,y_test,model):
    y_proba = model.predict_proba(X_test)[:, 1]
    # Split scores by class
    scores_good    = y_proba[y_test == 0]
    scores_default = y_proba[y_test == 1]
    # KS statistic
    ks_stat, p_value = ks_2samp(scores_good, scores_default)
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value:      {p_value:.4f}")
    return ks_stat, p_value


#gini coefficient evaluation on xgboost
def gini_coefficient(X_test,y_test,model):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    gini = 2 * roc_auc_score(y_test, y_pred_proba) - 1

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return fpr , tpr

#stratifield cross validation
def cross_validation(X_train,y_train,model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_train, y_train, 
                                cv=cv, scoring='roc_auc')
    #convert to auc to Gini
    gini_scores = 2 * auc_scores - 1
    return auc_scores, gini_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_cross_validation = cross_validation()
    data_gini_coefficient = gini_coefficient()
    data_kini_stat = plot_ks()

