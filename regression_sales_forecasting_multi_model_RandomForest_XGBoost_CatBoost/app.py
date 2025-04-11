
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🛍️ Sales Forecasting (Multi-Model)", layout="centered")
st.title("📈 Sales Forecasting with Model Selection")

model_choice = st.selectbox("Choose Forecasting Model", ["RandomForest", "XGBoost", "CatBoost"])
model_path = f"{model_choice.lower()}_sales_model.pkl"

try:
    model = joblib.load(model_path)
    features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("Please run model_training.py first to train models.")
    st.stop()

st.subheader("Upload Sales Data")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    input_df = df[features]
    predictions = model.predict(input_df)
    df["Predicted_Sales"] = predictions
    st.subheader("📊 Forecast Results")
    st.write(df[["Store", "Date", "Sales", "Predicted_Sales"]].head(10))
    st.download_button("Download Forecast", df.to_csv(index=False), "forecasted_sales.csv")
