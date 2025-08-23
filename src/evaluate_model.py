import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from .preprocessing import ALL_FEATURES, TARGET_COL

MODEL_PATH = 'models/churn_model.pkl'
DATA_PATH = 'data/Churn_Modelling.csv'

def evaluate():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    print('ROC-AUC', roc_auc_score(y, y_prob))
    print(classification_report(y, y_pred))

if __name__=="__main__":
    evaluate()