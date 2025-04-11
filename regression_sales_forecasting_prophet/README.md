
# Prophet-Based Sales Forecasting Project

## 📌 Business Use Case
This version of the Sales Forecasting project uses **Facebook Prophet**, a time series forecasting model, to predict weekly sales based on historical trends.

## 🔍 Features
- Data Aggregation by week
- Forecast with Prophet
- Streamlit interface to adjust forecast window and visualize trends

## 🛠 Tech Stack
- Python
- Prophet
- Streamlit
- Plotly

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## 📂 Files
- `sales_data.csv`: Input historical sales data
- `model_training.py`: Trains Prophet model
- `app.py`: Streamlit app to visualize forecast
