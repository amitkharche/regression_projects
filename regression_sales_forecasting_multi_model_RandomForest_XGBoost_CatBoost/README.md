
# 🧠 Multi-Model Sales Forecasting Project

This project demonstrates a modular approach to **forecasting retail sales** using **multiple machine learning regression models** — including **Random Forest, XGBoost, and CatBoost**. It allows for easy comparison of their performance through a **Streamlit web application**.

---

## 📌 Business Use Case

Retail businesses need accurate sales forecasts to make informed decisions about:
- **Inventory planning**
- **Workforce scheduling**
- **Promotion timing**
- **Revenue prediction**

This project helps decision-makers compare different algorithms to choose the most suitable model for their forecasting needs.

---

## 🔍 Problem Statement

> Build a predictive model to **forecast weekly sales** using a combination of:
- **Numerical features** (e.g., promotional discounts)
- **Categorical features** (e.g., store type, region)
- **Time-based features** (e.g., week of year, month)

The goal is to:
- Train and evaluate multiple ML models
- Compare performance using MAE, RMSE, and R²
- Serve results via a Streamlit app

---

## 🗂️ Project Structure

```
multi_model_sales_forecasting/
│
├── sales_data.csv                # Sample (synthetic) sales dataset
├── model_training.py            # Script to train RandomForest, XGBoost, and CatBoost
├── app.py                       # Streamlit app for interactive prediction
├── requirements.txt             # All dependencies
├── README.md                    # Project overview
└── models/
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    ├── catboost_model.pkl
    └── feature_columns.pkl      # Features used during model training
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/amitkharche/regression_projects.git
cd regression_projects/regression_sales_forecasting_multi_model_RandomForest_XGBoost_CatBoost
```

### 2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Models

```bash
python model_training.py
```

This script will:
- Read `sales_data.csv`
- Extract features
- Train 3 models: RandomForest, XGBoost, and CatBoost
- Save models to the `models/` folder

### 4️⃣ Launch Streamlit App

```bash
streamlit run app.py
```

---

## 💡 Features of Streamlit App

- Upload new sales data for prediction
- Choose which model to use (RandomForest, XGBoost, or CatBoost)
- See predictions and performance metrics in real-time
- Display visualizations of actual vs predicted sales

---

## 🧪 Models Used

| Model         | Library          | Key Traits                            |
|---------------|------------------|----------------------------------------|
| RandomForest  | `scikit-learn`   | Fast, robust, interpretable            |
| XGBoost       | `xgboost`        | Optimized gradient boosting, fast      |
| CatBoost      | `catboost`       | Handles categorical data natively      |

---

## 📊 Evaluation Metrics

Models are evaluated on:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

These metrics are printed in the console and shown on the Streamlit dashboard.

---

## 📁 Dataset Details

- **File**: `sales_data.csv`
- **Type**: Simulated retail sales dataset
- **Columns include**:
  - Store
  - Date
  - Promo
  - Sales

> **Note:** This dataset is synthetically generated and does not represent any real business data.

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [CatBoost](https://catboost.ai/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)

---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Amit Kharche**  
📬 Connect on [LinkedIn](https://www.linkedin.com/in/your-link/)  
📧 Email: amit@example.com

---

## 💬 Feedback

Have suggestions or improvements? Feel free to open an issue or pull request. Contributions are welcome!

---



