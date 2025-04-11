
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🛒 Sales Forecasting", layout="centered")
st.title("📈 Sales Forecasting App")

model = joblib.load("sales_model.pkl")
features = joblib.load("model_features.pkl")

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
    st.subheader("🔮 Forecast Results")
    st.write(df[["Store", "Date", "Sales", "Predicted_Sales"]].head(10))
    st.download_button("Download Predictions", df.to_csv(index=False), "predicted_sales.csv")
