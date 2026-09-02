import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

###columns needed for the encoding 
# Columns needed for imputation / encoding
MEAN_MISSING_COLS = [
    'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'OWN_CAR_AGE', 'EXT_SOURCE_1',
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'ELEVATORS_AVG',
    'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LIVINGAREA_AVG', 'APARTMENTS_MODE',
    'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LIVINGAREA_MODE', 'APARTMENTS_MEDI',
    'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LIVINGAREA_MEDI', 'TOTALAREA_MODE',
    'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 'AMT_CREDIT',
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_EMP_PHONE',
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY']
MOST_FREQUENT_COLS = ['OCCUPATION_TYPE']
ONE_HOT_COLS = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']
FREQUENCY_COLS = ['ORGANIZATION_TYPE']



#preprocess the data with information value to remove the columns with the low iv
def calculate_iv(X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    def calc(col: str) -> Tuple[float, float]:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > bins:
            binned = pd.qcut(pd.to_numeric(series, errors="coerce"), bins, duplicates="drop")
        else:
            binned = series
        g = y.groupby(binned, observed=True).agg(
            e=lambda x: (x == 1).sum(),
            n=lambda x: (x == 0).sum(),)
        g = (g / g.sum()).replace(0, np.nan).dropna()
        g["woe"] = np.log(g.e / g.n)
        return ((g.e - g.n) * g.woe).sum(), g.woe.mean()

    scores = {c: calc(c) for c in X.columns}
    return pd.DataFrame({
        "Feature": scores.keys(),
        "IV Score": [v[0] for v in scores.values()],
        "Avg WoE": [v[1] for v in scores.values()],
    }).round(4).sort_values("IV Score", ascending=False).reset_index(drop=True)


def get_low_iv_columns(X: pd.DataFrame, y: pd.Series, threshold: float = 0.02) -> List[str]:
    """Return column names with IV"""
    iv_table = calculate_iv(X, y)
    low_iv_cols = iv_table.loc[iv_table["IV Score"] <= threshold, "Feature"].tolist()
    logger.info(f"Identified {len(low_iv_cols)} low-IV columns to drop: {low_iv_cols}")
    return low_iv_cols

def drop_columns(X: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Drop the given columns"""
    return X.drop(columns=columns_to_drop, errors="ignore")

   

#winzorization for outlier capping 
def winsorize(X: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    """Clip values outside IQR-derived bounds computed from the lower/upper percentiles."""
    Q1 = np.percentile(X, lower * 100, axis=0)
    Q3 = np.percentile(X, upper * 100, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(X, lower_bound, upper_bound)


class NamedWinsorizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around `winsorize` that preserves column names."""

    def __init__(self, lower: float = 0.05, upper: float = 0.95):
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y=None) -> "NamedWinsorizer":
        self.feature_names_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            winsorize(X, self.lower, self.upper),
            columns=self.feature_names_,
            index=X.index,)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_names_)
    

#encoding the data and filling the missing data which includes column transformer
def build_preprocessing_pipeline():
    """Build the ColumnTransformer: impute, winsorize numeric, encode categorical."""
    mean_miss = Pipeline(steps=[
        ("mean_impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("winsorize", NamedWinsorizer()),
    ])

    most_freq = Pipeline(steps=[
        ("most_frequent_impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("categorical_frequency", CountFrequencyEncoder(encoding_method="count", missing_values="ignore")),
    ])

    freq_encoder = Pipeline(steps=[
        ("frequency", CountFrequencyEncoder(encoding_method="count", missing_values="ignore")),
    ])

    onehot_encoder = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("mean_missing", mean_miss, MEAN_MISSING_COLS),
        ("most_frequent", most_freq, MOST_FREQUENT_COLS),
        ("frequency_encoder", freq_encoder, FREQUENCY_COLS),
        ("onehot_encoder", onehot_encoder, ONE_HOT_COLS),
    ], remainder="passthrough")