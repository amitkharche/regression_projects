
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Feature & target split
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Data cleaning (remove rows with nulls if any)
X = X.dropna()
y = y.loc[X.index]

# Feature selection using SelectKBest
k_best_selector = SelectKBest(score_func=f_regression, k=5)

# Pipeline: Standardization + Feature Selection + Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', k_best_selector),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

# Save best model and selected features
joblib.dump(grid_search.best_estimator_, "regression_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("✅ Model training completed and saved!")