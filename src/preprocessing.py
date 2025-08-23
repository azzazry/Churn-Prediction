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