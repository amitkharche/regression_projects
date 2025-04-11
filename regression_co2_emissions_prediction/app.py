
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🚗 CO2 Emissions Estimation", layout="centered")
st.title("🌍 Predict CO2 Emissions from Vehicle Specs")

model = joblib.load("co2_model.pkl")
features = joblib.load("co2_features.pkl")

st.subheader("Upload Vehicle Data for Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    input_df = df[features]
    prediction = model.predict(input_df)
    df["Predicted_CO2_Emissions"] = prediction
    st.write(df.head())
    st.download_button("Download Predictions", df.to_csv(index=False), "predicted_co2.csv")
