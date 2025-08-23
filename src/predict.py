import pandas as pd
import joblib
from .preprocessing import ensure_feature_order

class ChurnPredictor:
    def __init__(self, model_path: str):
        self.model_path = joblib.load(model_path)
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_feature_order(df)
        proba = self.model.predict_proba(df)[:,1]
        label = self.model.predict(df)
        out = df.copy()
        out["Churn_Prob"] = proba
        out["Prediction_Label"] = label
        return out