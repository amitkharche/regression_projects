
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="⚡ Energy Consumption Prediction", layout="centered")
st.title("⚡ Predict Smart Grid Energy Consumption")

try:
    model = joblib.load("energy_model.pkl")
    features = joblib.load("energy_features.pkl")
except FileNotFoundError:
    st.error("Please run model_training.py first to train and save the model.")
    st.stop()

st.subheader("Upload Input Data for Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["DateTime"])
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    input_df = df[features]
    prediction = model.predict(input_df)
    df["PredictedEnergyConsumption"] = prediction
    st.subheader("🔋 Prediction Output")
    st.write(df[["DateTime", "PredictedEnergyConsumption"]].head(10))
    st.download_button("Download Results", df.to_csv(index=False), "predicted_energy.csv")
