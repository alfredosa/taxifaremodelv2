from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from TaxiFareModel.params import MODEL_NAME, BUCKET_NAME
from google.cloud import storage
from datetime import datetime
import joblib
import pandas as pd

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

@app.get("/predict")
def predict(pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    hardcoded = "2022-pred-1"
    print(pickup_datetime)
    formatted_pickup_datetime = datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d %H:%M:%S UTC")

    dict = {
        "key": [hardcoded],
        "pickup_datetime":[formatted_pickup_datetime],
        "pickup_longitude": [float(pickup_longitude)],
        "pickup_latitude": [float(pickup_latitude)],
        "dropoff_longitude": [float(dropoff_longitude)],
        "dropoff_latitude": [float(dropoff_latitude)],
        "passenger_count": [int(passenger_count)]
    }
    df = pd.DataFrame.from_dict(dict)
    # this would be via google

    # client = storage.Client().bucket(BUCKET_NAME)

    # storage_location = 'models/{}/v3/{}'.format(
    #     MODEL_NAME,
    #     'model.joblib')
    # blob = client.blob(storage_location)
    # blob.download_to_filename('model.joblib')

    # print("=> pipeline downloaded from storage")

    pipeline = joblib.load('model.joblib')

    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df)
    else:
        y_pred = pipeline.predict(df)
    print(y_pred)
    return {'fare': y_pred[0]}
