
import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly
import plotly.graph_objects as go


st.set_page_config(page_title="🛒 Sales Forecasting (Prophet)", layout="centered")
st.title("📈 Sales Forecasting with Prophet")

# Load model
model = joblib.load("prophet_sales_model.pkl")

# Forecast future
st.subheader("🔮 Forecast Settings")
periods = st.slider("Weeks to forecast", min_value=4, max_value=52, value=12)
future = model.make_future_dataframe(periods=periods, freq='W')
forecast = model.predict(future)

# Show forecast
st.subheader("📊 Forecasted Sales")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Download forecast
st.download_button("Download Forecast", forecast.to_csv(index=False), "sales_forecast.csv")
