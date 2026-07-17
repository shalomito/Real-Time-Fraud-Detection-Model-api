import pandas as pd
from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel

#Initialize the FastApi()

app = FastAPI(
    title ="Fraud detection model API",
    version = 'version 1'
)

#Load the model from the FDM folder
model_loads = joblib.load("my_fraud_detection_model.pkl")


#Create a Class for Transactions()

class CustomerTransaction(BaseModel):
    hour_of_the_day: int
    is_night_transaction: int
    merchant_category: str
    transaction_amount: float
    time_since_last_tnx_hrs: int
    failed_attempts: int
    pin_recently_changed: int
    transaction_hour: int


#Create the GET, home function
@app.get("/")
def home():
    return {'message': 'API is running..'}


#Create the POST, predict funtion
@app.post("/predict")
def predict(customer: CustomerTransaction):

    input_data = pd.DataFrame({
    'hour_of_the_day': [customer.hour_of_the_day],
    'is_night_transaction': [customer.is_night_transaction],
    'merchant_category': [customer.merchant_category],
    'transaction_amount': [customer.transaction_amount],
    'time_since_last_tnx_hrs': [customer.time_since_last_tnx_hrs],
    'failed_attempts': [customer.failed_attempts],
    'pin_recently_changed': [customer.pin_recently_changed],
    'transaction_hour': [customer.transaction_hour]
})

    predictions_true_false = model_loads.predict(input_data)
    prediction_percent_prob = model_loads.predict_proba(input_data)[0, 1]

    if prediction_percent_prob > 0.7:

        return{
            'Result' : 'FRAUD',
            'Probability' : prediction_percent_prob*100
        }
    else:
        return{
            'Result': 'NOT FRAUD',
            'Probability' : prediction_percent_prob*100
        }


