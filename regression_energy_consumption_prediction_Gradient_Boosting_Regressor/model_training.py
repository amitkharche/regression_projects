
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("energy_data.csv", parse_dates=["DateTime"])

# Extract time-based features
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

X = df[["Temperature", "Humidity", "ApplianceUsage", "LightUsage", "Occupancy", "Hour", "DayOfWeek"]]
y = df["EnergyConsumption"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with scaling and Gradient Boosting
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(pipeline, "energy_model.pkl")
joblib.dump(X.columns.tolist(), "energy_features.pkl")
