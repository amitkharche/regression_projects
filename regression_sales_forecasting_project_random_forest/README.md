
# Sales Forecasting Regression Project

## 📌 Business Use Case
Forecasting weekly sales for retail stores using historical data, seasonality, and promotions to optimize inventory and improve revenue planning.

## 🧠 Problem Statement
Given historical weekly sales, promotions, and store information, build a model that can predict future sales.

## 📦 Features
- Categorical encoding (Store)
- Time-based features (Month, Week, Year)
- Hyperparameter tuning with GridSearchCV
- RandomForestRegressor pipeline
- Interactive Streamlit dashboard

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 📂 Files
- `sales_data.csv`: Synthetic historical sales data
- `model_training.py`: Trains and saves the model
- `app.py`: Streamlit interface for forecasting
