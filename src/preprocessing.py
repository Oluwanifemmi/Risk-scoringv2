#import the data and preproces the data for feature engineering

#libraries to import the data
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split



#import the dataset
logger = logging.getLogger(__name__)
def data_import(datapath:str):
    try:
        data = pd.read_csv(datapath)
    except FileNotFoundError as e:
        logger.error(f"Could not find data file at {datapath}")
        raise e

    logger.info(f"Loaded data with shape {data.shape} from {datapath}")
    return data
    


#data splitting 
def data_split(data: pd.DataFrame,target_col: str = "TARGET",test_size: float = 0.33,random_state: int = 42,):
    if target_col not in data.columns:
        raise KeyError(f"'{target_col}' not found in data columns")
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info(f"Split data: X_train {X_train.shape}, X_test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test