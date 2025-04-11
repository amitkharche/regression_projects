
# California Housing Advanced Regression Project

## 📌 Business Context
This enhanced version uses the open-source California Housing dataset with advanced regression techniques to predict median house prices. It includes data cleaning, feature selection, and hyperparameter tuning to create a robust model.

## 🧠 Problem Statement
Estimate median house values using predictive modeling with optimized performance for business insights and housing trend forecasting.

## 🔍 Features Implemented
- Data cleaning
- Standardization
- Feature selection (SelectKBest)
- Hyperparameter tuning (GridSearchCV)
- Random Forest Regressor

## 🛠️ Tech Stack
- Python
- scikit-learn
- joblib
- pandas
- Streamlit

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 📂 Files
- `model_training.py` — Preprocesses, tunes, and saves the trained model
- `app.py` — Streamlit dashboard
- `regression_model.pkl` — Trained pipeline
- `model_features.pkl` — Original column names
