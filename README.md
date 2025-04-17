
# 🧠 Regression Projects Portfolio

Welcome to my collection of **regression modeling projects**, where I explore various machine learning techniques to solve real-world business forecasting problems. Each project is fully modular, documented, and includes a user interface powered by **Streamlit** for interactive model exploration.

---

## 📁 Repository Structure

Each sub-folder within this repository represents a standalone regression project with its own dataset, model training code, Streamlit interface, and detailed README.

```
regression_projects/
├── regression_sales_forecasting_multi_model_RandomForest_XGBoost_CatBoost/
├── regression_sales_forecasting_prophet/
├── regression_energy_consumption/
└── ...
```

---

## 🚀 Projects Overview

### 🔷 [Sales Forecasting with RF/XGB/CatBoost](./regression_sales_forecasting_multi_model_RandomForest_XGBoost_CatBoost)
- Forecast weekly sales using tree-based regressors
- Compare performance of RandomForest, XGBoost, and CatBoost
- Includes preprocessing, hyperparameter tuning, and Streamlit dashboard

### 🔮 [Sales Forecasting with Facebook Prophet](./regression_sales_forecasting_prophet)
- Time series forecasting using Facebook Prophet
- Weekly data aggregation, trend/seasonality decomposition
- Interactive forecast tuning via Streamlit

### ⚡ [Energy Consumption Prediction](./regression_energy_consumption)
- Predict appliance energy usage using GradientBoostingRegressor
- Time-based feature engineering (hour, day of week)
- Integrated pipeline and model export

---

## 🛠 Tech Stack

- Python 3.8+
- Scikit-learn, XGBoost, CatBoost, Prophet
- Pandas, NumPy, Matplotlib, Plotly
- Streamlit for interactive UI
- Joblib for model serialization

---

## 📦 How to Use

### 🔧 Clone the Repository

```bash
git clone https://github.com/amitkharche/regression_projects.git
cd regression_projects
```

### 📂 Navigate to any Project

```bash
cd regression_sales_forecasting_prophet
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🏋️‍♂️ Train Model

```bash
python model_training.py
```

### 🌐 Launch Streamlit App

```bash
streamlit run app.py
```

---

## 📬 Contact

**Amit Kharche**  
🔗 [LinkedIn](https://www.linkedin.com/in/amitkharche)

---

## 📄 License

This repository is open-sourced under the MIT License.  
Feel free to fork, clone, and contribute!

---

## 🙌 Feedback & Contributions

If you found these projects helpful or have ideas for improvement, feel free to open an issue or a pull request.  
Let’s collaborate to build better data-driven solutions!

