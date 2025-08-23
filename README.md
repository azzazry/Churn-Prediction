# Churn Prediction API with FastAPI
The goal: an easy-to-use REST API that integrates predictive analytics into business workflows, helping decision-makers act before a customer leaves.

## 1) Setup

```bash
python -m venv .venv

# Windows
.venv/Scripts/activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
````

## 2) Data

Place `data/Churn_Modelling.csv` in the `data/` folder.

## 3) Train

```bash
python -m src.preprocessing
python -m src.train_model
python -m src.evaluate_model
```

Artifacts:

- `models/churn_model.pkl`
- `models/metrics.json`
- `models/test_predictions.csv`

## 4) Serve API

```bash
uvicorn app.main:app --reload
```

## 5) Example Request with JSON

```bash
[
  {
    "Username": "Aaz Zazri Nugraha",
    "CreditScore": 820,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 22,
    "Tenure": 7,
    "Balance": 120000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 6000.0
  }
]
```
