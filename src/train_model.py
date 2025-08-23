import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from .preprocessing import (
    build_preprocessor,
    ALL_FEATURES,
    TARGET_COL
)

DATA_PATH = 'data/Churn_Modelling.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
TEST_PRED_PATH = os.path.join(MODEL_DIR, 'test_prediction.csv')

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build Pipeline
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
        class_weight=None
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ]
    )
    
    # Train
    pipeline.fit(X_train,y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    
    metrics = {
        "roc_auc": auc,
        "precision_0": report['0']['precision'],
        "recall_0": report['0']['recall'],
        "precision_1": report['1']['precision'],
        "recall_1": report['1']['recall'],
        "accuracy": report['accuracy'],
        "macro_f1": report['macro avg']['f1-score'],
        "weighted_f1": report['weighted avg']['f1-score']        
    }
    
    print(json.dumps(metrics, indent=2))
    
    # Save Artifacts
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save Test Prediction
    test_out = X_test.copy()
    test_out['y_true'] = y_test.values
    test_out['y_prob'] = y_prob
    test_out['y_pred'] = y_pred
    test_out.to_csv(TEST_PRED_PATH, index=False)
    
if __name__=="__main__":
    main()