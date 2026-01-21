from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from pycaret.regression import load_model

app = FastAPI(title="AI Model API", version="1.0")

# Wczytanie modelu
with open("projekt/AutoML_pipeline/data/models/apartment_price_model.pkl", "rb") as f:
    model = load_model("projekt/AutoML_pipeline/data/models/apartment_price_model")
    print("Model loaded correctly")
    print(type(model))

# Definicja danych wejściowych
class InputData(BaseModel):
    type: str
    squareMeters: float
    rooms: int
    floor: int
    floorCount: int
    buildYear: int
    latitude: float
    longitude: float
    centreDistance: float
    poiCount: int
    ownership: str
    hasParkingSpace: str
    hasBalcony: str
    hasElevator: str
    hasSecurity: str
    hasStorageRoom: str
    isSchoolNear: int
    isClinicNear: int
    isPostOfficeNear: int
    isKindergartenNear: int
    isRestaurantNear: int
    isCollegeNear: int
    isPharmacyNear: int
    building_age: int
    floor_ratio: float
    sqm_per_room: float

@app.get("/")
def home():
    return {"message": "Model API działa poprawnie"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
