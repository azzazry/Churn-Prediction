````md
# Churn Project – Real-World DS

## 1) Setup

```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
````

## 2) Data

Place `data/Churn_Modelling.csv` in the `data/` folder.

## 3) Train

```bash
python -m src.train_model
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
    "CreditScore": 650,
    "Geography": "Indonesia",
    "Gender": "Male",
    "Age": 22,
    "Tenure": 3,
    "Balance": 120000.50,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
  },
  {
    "Username": "Nisa Muthmainnah",
    "CreditScore": 720,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 23,
    "Tenure": 5,
    "Balance": 80000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 60000.0
  }
]

```