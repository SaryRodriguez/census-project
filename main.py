from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from model.train_model import process_data, inference, CAT_FEATURES

app = FastAPI()

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 280464,
                "education": "Some-college",
                "education-num": 10,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "Black",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 80,
                "native-country": "United-States"
            }
        }

@app.get("/")
def root():
    return {"message": "Welcome to the Census Salary Prediction API!"}

@app.post("/predict")
def predict(data: CensusData):
    input_dict = {
        "age": data.age, "workclass": data.workclass, "fnlgt": data.fnlgt,
        "education": data.education, "education-num": data.education_num,
        "marital-status": data.marital_status, "occupation": data.occupation,
        "relationship": data.relationship, "race": data.race, "sex": data.sex,
        "capital-gain": data.capital_gain, "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week, "native-country": data.native_country,
        "salary": "<=50K"  # placeholder
    }
    df = pd.DataFrame([input_dict])
    X, _, _, _ = process_data(df, CAT_FEATURES, "salary", training=False, encoder=encoder)
    pred = inference(model, X)[0]
    label = ">50K" if pred == 1 else "<=50K"
    return {"prediction": label}