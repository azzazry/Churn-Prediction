# 📊 Churn Prediction API with FastAPI

**Goal:**
Menyediakan REST API yang mudah digunakan untuk melakukan prediksi *customer churn*, sehingga bisnis dapat mengambil tindakan sebelum pelanggan pergi.

---

## 🚀 1) Setup Environment

```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 2) Dataset

Letakkan file dataset berikut ke dalam folder `data/`:

```
data/Churn_Modelling.csv
```

---

## 🏋️ 3) Training Pipeline

Jalankan preprocessing, training, dan evaluasi model:

```bash
python -m src.preprocessing
python -m src.train_model
python -m src.evaluate_model
```

**Artifacts yang dihasilkan:**

* `models/churn_model.pkl` – Model yang sudah dilatih
* `models/metrics.json` – Metrik evaluasi model
* `models/test_predictions.csv` – Hasil prediksi pada data uji

---

## 🌐 4) Menjalankan API

```bash
uvicorn app.main:app --reload
```

API akan berjalan di:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧩 5) Contoh Request (JSON)

Gunakan *POST request* ke endpoint `/predict/`:

```json
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

---

## 📌 Endpoint Utama

| Endpoint    | Method | Deskripsi                      |
| ----------- | ------ | ------------------------------ |
| `/predict/` | POST   | Prediksi churn dari input JSON |

---

## 🛠 Tech Stack

* **FastAPI** – Framework backend untuk API
* **Scikit-learn** – Model machine learning
* **Pandas & NumPy** – Data processing
* **Uvicorn** – ASGI server
