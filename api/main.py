from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.Feature_engineering import build_preprocessing_pipeline


import sys
sys.path.append('/app')

app = FastAPI()

model = joblib.load('model/riskscore.pkl')


class creditfeature(BaseModel):
    CODE_GENDER: str
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: float
    FLAG_EMP_PHONE: int
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: str
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    OCCUPATION_TYPE: str
    APARTMENTS_AVG: float
    ELEVATORS_AVG: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    LIVINGAREA_AVG: float
    APARTMENTS_MODE: float
    FLOORSMAX_MODE: float
    FLOORSMIN_MODE: float
    LIVINGAREA_MODE: float
    APARTMENTS_MEDI: float
    FLOORSMAX_MEDI: float
    FLOORSMIN_MEDI: float
    LIVINGAREA_MEDI: float
    TOTALAREA_MODE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: int


@app.post("/predict_probability")
def predict_probability(data: creditfeature):
    input_data = pd.DataFrame([data.model_dump()])

    probability = float(model.predict_proba(input_data)[0][1])
    predicted_default = int(model.predict(input_data)[0])

    if probability < 0.35:
        risk_tier = "Low Risk"
    elif probability < 0.60:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "High Risk"

    return {
        'probability_default': round(probability, 6),
        'predicted_default': predicted_default,
        'risk_tier': risk_tier
    }