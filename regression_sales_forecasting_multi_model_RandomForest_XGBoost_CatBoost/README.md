
# Multi-Model Sales Forecasting Project

## 📌 Business Use Case
Forecast retail sales using different regression models — RandomForest, XGBoost, and CatBoost — and compare results.

## 🧠 Problem Statement
Use time-based and categorical features to predict weekly store sales and enable model comparison.

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 📦 Files
- `sales_data.csv` — Sample sales data
- `model_training.py` — Trains three models and saves them
- `app.py` — Streamlit interface for uploading data and choosing a model
