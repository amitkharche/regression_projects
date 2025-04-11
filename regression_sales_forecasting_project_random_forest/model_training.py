
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Load data
df = pd.read_csv("sales_data.csv", parse_dates=["Date"])

# Feature Engineering
df["Week"] = df["Date"].dt.isocalendar().week
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Prepare features and target
X = df[["Store", "Promo", "Week", "Month", "Year"]]
y = df["Sales"]

# Encode categorical features
categorical_features = ["Store"]
numeric_features = ["Promo", "Week", "Month", "Year"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid.fit(X_train, y_train)

# Save model and features
joblib.dump(grid.best_estimator_, "sales_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
