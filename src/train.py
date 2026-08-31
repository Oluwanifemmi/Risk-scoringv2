from pipelineforfeature_engineering import preprocess
from custom_transformers import winsorize, NamedWinsorizer
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV

def pipeline_xgboost(X_train,y_train):
    pipe = imbpipeline(steps=[
    ('ColumnTransformer', preprocess),
    ('OutlierCapping', NamedWinsorizer(lower=0.05, upper=0.95)),  
    ('SMOTE', SMOTE(sampling_strategy='minority', random_state=42)),
    ('SCALER', StandardScaler()),                                  
    ('xgboost', XGBClassifier(scale_pos_weight=93362/8117,
                              eta = 0.1, 
                              depth= 10, 
                              subsample=1.0, 
                              min_child_weight = 5,
                              col_sample_bytree = 0.8,
                              objective = 'binary:logistic'))
     ])


    #hyperparameter tunning
    grid = {
    'XGBOOST__learning_rate'  : [0.01, 0.05, 0.1, 0.3],
    'XGBOOST__n_estimators'   : [100, 300, 500],         
    'XGBOOST__max_depth'      : [4, 6, 8, 10],
    'XGBOOST__subsample'      : [0.6, 0.7, 0.8, 1.0],   
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
        random_state=42
    )
    search.fit(X_sample, y_sample)
    pipe = search.best_estimator_

    pipe.set_output(transform="pandas")
    return ("f Xgboost model training {pipe.fit(X_train,y_train}")