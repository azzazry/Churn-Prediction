from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

MODEL_PATH = 'models/churn_model.pkl'

app = FastAPI(title='Churn Prediction Service', version='1.0.0')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f'[WARN] Failed to load model from {model}: {e}')

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
    
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model/info")
async def model_info():
    if model is None:
        return {"model": None}
    steps = list(getattr(model, "steps", []))
    return {
        "type": str(type(model)),
        "has_prob": hasattr(model, "predict_proba"),
        "steps": [name for name, _ in steps]
    }

@app.post('/predict')
async def predict(data: List[CustomerData]):
    if model is None:
        return {"error": "Model not loaded. Train first or check model path."}
    
    df = pd.DataFrame([d.dict() for d in data])
    
    expected_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df[expected_cols]
    
    try:
        probs = model.predict_proba(df)[:,1]
        labels = model.predict(df)
    except NotFittedError:
        return {"error": "Model is not fitted. Retrain the model."}
    
    out = []
    for i, item in enumerate(data):
        out.append({
            "Username": item.Username,
            "Churn_Prob": float(probs[i]),
            "Prediction_Label": int(labels[i])
        })
    
    return {"prediction": out}