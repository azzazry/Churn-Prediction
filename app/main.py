from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

MODEL_PATH = 'models/churn_model.pkl'

app = FastAPI()
model = joblib.load(MODEL_PATH)

class CustomerData(BaseModel):
    Username: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.post('/predict')
async def predict(data: List[CustomerData]):
    df = pd.DataFrame([item.dict() for item in data])
    usernames = df.pop("Username")
    
    try:
        probs = model.predict_proba(df)[:, 1]
        labels = model.predict(df)
    except NotFittedError:
        return {"error": "Model is not fitted. Retrain the model."}

    predictions = [
        {"Username": usernames.iloc[i], "Churn_Prob": float(probs[i]), "Prediction_Label": int(labels[i])}
        for i in range(len(df))
    ]
    return {"prediction": predictions}