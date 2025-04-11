
import pandas as pd
from prophet import Prophet
import joblib

# Load data
df = pd.read_csv("sales_data.csv", parse_dates=["Date"])

# Aggregate by date across all stores for simplicity
df_agg = df.groupby("Date").agg({"Sales": "sum"}).reset_index()
df_agg.columns = ["ds", "y"]

# Train Prophet model
model = Prophet()
model.fit(df_agg)

# Save model
joblib.dump(model, "prophet_sales_model.pkl")
