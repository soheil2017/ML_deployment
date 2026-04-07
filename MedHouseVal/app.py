# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

model = joblib.load("model.joblib")

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                      features.AveBedrms, features.Population,
                      features.AveOccup, features.Latitude, features.Longitude]])
    prediction = model.predict(data)[0]
    return {"predicted_price": prediction}
