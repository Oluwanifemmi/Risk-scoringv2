import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from data_split import X, y


###columns needed for the encoding 
mean_missing = ['AMT_ANNUITY','AMT_GOODS_PRICE','OWN_CAR_AGE','EXT_SOURCE_1',
               'EXT_SOURCE_2','EXT_SOURCE_3','APARTMENTS_AVG','ELEVATORS_AVG',                 
                'FLOORSMAX_AVG','FLOORSMIN_AVG','LIVINGAREA_AVG','APARTMENTS_MODE',
                'FLOORSMAX_MODE','FLOORSMIN_MODE','LIVINGAREA_MODE','APARTMENTS_MEDI',
                'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI','LIVINGAREA_MEDI','TOTALAREA_MODE',
                'DAYS_LAST_PHONE_CHANGE','FLAG_DOCUMENT_3','AMT_CREDIT', 'REGION_POPULATION_RELATIVE',
                'DAYS_BIRTH', 'DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_EMP_PHONE',
                 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY' ]
most_frequent = ['OCCUPATION_TYPE']
one_hot = ['CODE_GENDER','NAME_INCOME_TYPE',
           'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS']
frequency = ['ORGANIZATION_TYPE']



#preprocess the data with information value to remove the columns with the low iv
def iv(X, y, bins=10):
    def calc(col):
        f = pd.qcut(pd.to_numeric(X[col], errors="coerce"), bins, duplicates="drop") if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > bins else X[col]
        g = X.groupby(f, observed=True)[y].agg(e=lambda x: (x==1).sum(), n=lambda x: (x==0).sum())
        g = (g/g.sum()).replace(0, np.nan).dropna()
        g["woe"] = np.log(g.e / g.n)
        return ((g.e - g.n) * g.woe).sum(), g.woe.mean()
    scores = {c: calc(c) for c in X.columns if c != y}
    return pd.DataFrame({"Feature": scores.keys(), "IV Score": [v[0] for v in scores.values()], "Avg WoE": [v[1] for v in scores.values()]}).round(4).sort_values("IV Score", ascending=False).reset_index(drop=True)



def drop_low_iv_columns(X,y):
    result = iv(X,'y')
    dropcolumn = result[result["IV Score"] <= 0.02]["Feature"].tolist()
    return X.drop(dropcolumn)
    
    
   

#winzorization for outlier capping 
def winsorize(X, lower=0.25, upper=0.75):
    Q1 = np.percentile(X, lower * 100, axis=0)
    Q3 = np.percentile(X, upper * 100, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(X, lower_bound, upper_bound)


class NamedWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.05, upper=0.95):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.feature_names_ = list(X.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(winsorize(X, self.lower, self.upper),
                            columns=self.feature_names_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    

#encoding the data and filling the missing data 
def encoding_imputing(X):
    mean_miss = Pipeline(steps=[
    ("MeanFormissing",SimpleImputer(missing_values=np.nan,strategy='mean'))])

    most_freq = Pipeline(steps=[
    ("Mostfrequentformissing",SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('categorical_fre',CountFrequencyEncoder(encoding_method='count', missing_values='ignore'))])

    freq_encoder =Pipeline(steps=[("frequency",CountFrequencyEncoder(encoding_method='count',
                                     missing_values='ignore'))])

    onehot_encoder =Pipeline(steps=[
                         ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

    preprocess = ColumnTransformer([
    ('MeanFormissing',         mean_miss,      mean_missing),
    ('Mostfrequentformissing', most_freq,      most_frequent),
    ('FrequencyEncoder',       freq_encoder,   frequency),
    ('Onehotencoder',          onehot_encoder, one_hot)],remainder='passthrough')

    return preprocess