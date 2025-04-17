
# 📈 Sales Forecasting Regression Project

This project demonstrates how to build a machine learning pipeline for **forecasting weekly sales** of retail stores using **RandomForestRegressor**, time-based feature engineering, and a **Streamlit** dashboard for interactive predictions.

---

## 📌 Business Use Case

Retail businesses rely heavily on sales forecasts for:
- 📦 Inventory management
- 📅 Promotion planning
- 💰 Revenue projections

Accurate weekly forecasts help avoid **stockouts**, reduce **overstocking**, and **maximize profit**. This project enables stakeholders to visualize and plan based on predictive insights.

---

## 🧠 Problem Statement

> Build a machine learning model that predicts **future weekly sales** based on:
- Store ID
- Dates
- Promotional events
- Seasonal patterns (Month, Week, Year)

The model uses historical data to learn trends and patterns that influence retail performance.

---

## 📊 Features & Techniques

| Feature | Description |
|--------|-------------|
| 🏪 Categorical Encoding | One-hot encoding of `Store` |
| 📆 Time-based Features | Extracted `Month`, `Week`, `Year` from `Date` |
| 🧪 Model | `RandomForestRegressor` from `scikit-learn` |
| 🔍 Tuning | `GridSearchCV` used to optimize hyperparameters |
| ⚙️ Pipeline | End-to-end `Pipeline` with scaler + model |
| 🌐 Interface | Interactive `Streamlit` dashboard for prediction |

---

## 📂 Project Structure

```
sales_forecasting_project/
│
├── sales_data.csv          # Synthetic historical sales dataset
├── model_training.py       # Script for training the model
├── app.py                  # Streamlit UI for predictions
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── models/
    ├── sales_model.pkl     # Saved RandomForest model pipeline
    └── feature_columns.pkl # Feature list used for training
```

---

## 🧪 Dataset Description

- **File**: `sales_data.csv`
- **Type**: Synthetic, generated dataset
- **Columns**:
  - `Date` — Weekly date
  - `Store` — Store ID (categorical)
  - `Promo` — Promotion flag (0/1)
  - `Sales` — Weekly sales value (target)

> ⚠️ **Note:** This is **synthetic data** and does not represent any real business or customer data.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/amitkharche/regression_projects.git
cd regression_projects/regression_sales_forecasting_Store_Promo_RandomForest
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python model_training.py
```

This will:
- Load `sales_data.csv`
- Engineer features
- Train a `RandomForestRegressor` pipeline
- Save the model and feature columns to the `models/` directory

### 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run app.py
```

You can now:
- Upload new sales data
- View model predictions
- Explore trends through interactive charts

---

## 📉 Evaluation Metrics

During training, the model is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

These metrics are printed in the console for reference.

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Matplotlib](https://matplotlib.org/)

---

## 🙋‍♂️ Author

**Amit Kharche**  
📬 [LinkedIn](https://www.linkedin.com/in/amitkharche)  

---

## 📬 Feedback

Have feedback or ideas to improve?  
Open an issue or submit a pull request. Contributions are welcome!

---


- `app.py`: Streamlit interface for forecasting
