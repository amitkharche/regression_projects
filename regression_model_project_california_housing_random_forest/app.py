
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Linear Regression Model", layout="centered")
st.title("🏡 California Housing Price Prediction")

model = joblib.load("regression_model.pkl")
features = joblib.load("model_features.pkl")

st.subheader("Enter Input Values")
input_data = {}
for col in features:
    input_data[col] = st.number_input(f"{col}", value=0.0)

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
st.success(f"🏠 Predicted Median House Value: ${prediction * 100000:,.2f}")
