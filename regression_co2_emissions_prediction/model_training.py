
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("co2_emissions.csv")

X = df.drop(columns=["CO2Emissions"])
y = df["CO2Emissions"]

numeric_features = ["EngineSize", "Cylinders", "FuelConsumption_City", "FuelConsumption_Hwy", "FuelConsumption_Comb"]
categorical_features = ["FuelType"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
#print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Save model and features
joblib.dump(pipeline, "co2_model.pkl")
joblib.dump(X.columns.tolist(), "co2_features.pkl")
print("model training done")

