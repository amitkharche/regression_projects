
# Energy Consumption Prediction (Smart Grid)

## 💡 Business Problem
Accurately predict hourly energy consumption to optimize energy distribution and reduce grid load for smart grid planning.

## 🧠 Solution
This project uses historical consumption patterns, weather, appliance and lighting usage, and occupancy to train a model that forecasts energy usage.

## 🛠 Tech Stack
- Python
- scikit-learn (Gradient Boosting Regressor)
- Streamlit
- pandas, joblib

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 📂 Files
- `energy_data.csv`: Hourly smart home energy consumption (simulated)
- `model_training.py`: Preprocesses and trains GradientBoostingRegressor
- `app.py`: Streamlit interface for upload + prediction
