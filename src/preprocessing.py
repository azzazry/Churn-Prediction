from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary'
]

CAT_COLS = [
    'Geography',
    'Gender'
]

PASSTHROUGH_COLS = [
    'HasCrCard',
    'IsActiveMember'
]

ALL_FEATURES = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

TARGET_COL = 'Exited'

def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop=None), CAT_COLS),
        ('bin', 'passthrough', PASSTHROUGH_COLS)
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    return preprocessor

def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    safe_columns = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 5,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }
    out = pd.DataFrame()
    for col in ALL_FEATURES:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = safe_columns[col]
    return out