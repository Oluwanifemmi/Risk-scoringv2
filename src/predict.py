import logging
from typing import Tuple
import pandas as pd
import joblib

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



# def predict_model(X_test,model):
#    prediction = model.predict_proba(X_test)[:,1][10]
#    return prediction

def predict_default(model, X):
    prob       = model.predict_proba(X)[:, 1]
    prediction = model.predict(X)

    results = pd.DataFrame({
        'index'               : X.index,
        'probability_default' : prob,
        'predicted_default'   : prediction,
        'risk_tier'           : pd.cut(
                                    prob,
                                    bins=[0, 0.3, 0.6, 1.0],
                                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                                )
    })
    
    return results.sort_values('probability_default', ascending=False)


if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO)

   model = joblib.load(MODEL_PATH)
   X_train, X_test, y_train, y_test = load_processed_data()

   the_prediction = predict_default(model, X_test)