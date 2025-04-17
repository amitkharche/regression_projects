

# model_training.py

"""
🛠️ Model Training Script: Prophet-Based Sales Forecasting

This script loads synthetic weekly sales data, performs preprocessing,
and trains a Facebook Prophet model to forecast future sales.

Features:
- Weekly data aggregation
- Prophet model training with additive seasonality
- CLI-based forecast period customization
- Optional forecast CSV/Plot export
- Logging and config-based flexibility

Author: Amit Kharche
LinkedIn: https://www.linkedin.com/in/amitkharche
"""

import pandas as pd
from prophet import Prophet
import joblib
import argparse
import os
import logging
from prophet.plot import plot_plotly
import plotly.io as pio

# ---------------------------
# 📋 Setup logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# ⚙️ CLI Argument Parsing
# ---------------------------
parser = argparse.ArgumentParser(description="Train a Prophet model for sales forecasting.")
parser.add_argument("--data_path", type=str, default="sales_data.csv", help="Path to input CSV file")
parser.add_argument("--model_path", type=str, default="models/prophet_sales_model.pkl", help="Where to save the model")
parser.add_argument("--periods", type=int, default=12, help="Weeks to forecast for preview")
parser.add_argument("--save_forecast", action="store_true", help="Save forecast to CSV")
parser.add_argument("--save_plot", action="store_true", help="Save forecast plot as PNG")
args = parser.parse_args()

# ---------------------------
# 📂 Load and Validate Data
# ---------------------------
logging.info("🔄 Loading sales data...")
try:
    df = pd.read_csv(args.data_path, parse_dates=["Date"])
    logging.info(f"✅ Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    raise Exception(f"❌ Could not find the file: {args.data_path}")

# ---------------------------
# 📊 Aggregate Sales by Date
# ---------------------------
df_agg = df.groupby("Date").agg({"Sales": "sum"}).reset_index()
df_agg.columns = ["ds", "y"]

logging.info("📅 Aggregated weekly sales. Sample:")
logging.info(f"\n{df_agg.head()}")

# ---------------------------
# 🔮 Train Prophet Model
# ---------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive'
)

logging.info("🚀 Training Prophet model...")
model.fit(df_agg)
logging.info("✅ Model training complete.")

# ---------------------------
# 💾 Save Model
# ---------------------------
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
joblib.dump(model, args.model_path)
logging.info(f"💾 Model saved to: {args.model_path}")

# ---------------------------
# 📈 Optional: Generate Forecast
# ---------------------------
logging.info(f"📈 Generating forecast for {args.periods} weeks...")
future = model.make_future_dataframe(periods=args.periods, freq='W')
forecast = model.predict(future)

if args.save_forecast:
    forecast_path = os.path.join(os.path.dirname(args.model_path), "sample_forecast.csv")
    forecast.to_csv(forecast_path, index=False)
    logging.info(f"📁 Forecast saved to: {forecast_path}")

if args.save_plot:
    fig = plot_plotly(model, forecast)
    plot_path = os.path.join(os.path.dirname(args.model_path), "forecast_plot.png")
    fig.write_image(plot_path)
    logging.info(f"🖼️ Forecast plot saved to: {plot_path}")

logging.info("✅ Script completed successfully.")

