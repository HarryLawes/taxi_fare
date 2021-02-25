from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from google.cloud import storage
from TaxiFareModel.predict import download_model

BUCKET_NAME='wagon-ml-lawes-00'
STORAGE_LOCATION = 'models/taxifare/model.joblib'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare/")
def predict(key,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    
    X_test = pd.DataFrame(dict(
        key=[key],
        pickup_datetime=[pickup_datetime],
        pickup_longitude=[float(pickup_longitude)],
        pickup_latitude=[float(pickup_latitude)],
        dropoff_longitude=[float(dropoff_longitude)],
        dropoff_latitude=[float(dropoff_latitude)],
        passenger_count=[int(passenger_count)]))

    pipeline = joblib.load("model.joblib")
    #pipeline = download_model()
    y_pred = pipeline.predict(X_test)

    return {'prediction': float(y_pred[0])}