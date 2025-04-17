
# 🔮 Prophet-Based Sales Forecasting Project

This project utilizes **Facebook Prophet**, a powerful time series forecasting model, to **predict weekly retail sales** based on historical patterns and trends. Designed for simplicity and scalability, the model integrates into a **Streamlit app** for interactive forecasting and visualization.

---

## 📌 Business Use Case

Retailers require accurate sales forecasting for:
- 📦 Efficient inventory management
- 💰 Strategic revenue planning
- 📅 Aligning marketing and promotional campaigns

By leveraging **seasonality**, **trend analysis**, and **historical sales behavior**, this project provides a time-aware forecast of weekly retail sales using **Facebook Prophet**.

---

## 🧠 Problem Statement

> Build a time series forecasting model that can learn from historical weekly sales data and predict future sales using Prophet.

The dataset includes:
- Weekly sales figures
- Promotion indicators
- Store-level information

We aggregate the data at the **weekly level** and use Prophet to model trends, seasonality, and changepoints in the sales trajectory.

---

## 🔍 Features

| Feature | Description |
|--------|-------------|
| 📆 Weekly Aggregation | Data is grouped by `Date` to compute weekly total sales |
| 🔮 Prophet Model | Forecasts based on trends, seasonality, and holidays |
| 📈 Streamlit App | Interactive dashboard to explore forecast with adjustable parameters |
| 🎛️ User Controls | Choose forecast horizon and view trend decomposition |
| 📊 Plotly Charts | Dynamic time series visualization and forecast plots |

---

## 📂 Project Structure

```
regression_sales_forecasting_prophet/
│
├── sales_data.csv          # Synthetic historical weekly sales dataset
├── model_training.py       # Script to prepare data and train Prophet model
├── app.py                  # Streamlit app to visualize forecasts
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## 🧪 Dataset Description

- **File**: `sales_data.csv`
- **Type**: Synthetic retail sales data
- **Columns**:
  - `Date` — Weekly date
  - `Store` — Store ID (categorical)
  - `Promo` — Promotion flag (0/1)
  - `Sales` — Weekly sales figure

> ⚠️ This dataset is synthetically generated and is used for demonstration purposes only.

---

## 🛠 Tech Stack

- [Python](https://www.python.org/)
- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Pandas](https://pandas.pydata.org/)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/amitkharche/regression_projects.git
cd regression_projects/regression_sales_forecasting_prophet
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Forecasting Model

```bash
python model_training.py
```

This step:
- Loads `sales_data.csv`
- Aggregates data by week
- Trains the Prophet model
- Prepares future forecast dataframe

### 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run app.py
```

From here, you can:
- Adjust the forecast horizon (e.g., 30, 60, 90 days)
- Visualize forecasted sales with confidence intervals
- View trend and seasonality decomposition

---

## 📉 Output: Forecast Visuals

- 📈 Sales forecast with confidence intervals
- 🌀 Weekly and yearly seasonality
- 🔄 Interactive controls for tuning forecast horizon

---

## 🙋‍♂️ Author

**Amit Kharche**  
📬 [LinkedIn](https://www.linkedin.com/in/amitkharche)  

---

## 📬 Feedback

Feel free to suggest improvements or open issues. Contributions are always welcome!

---

## 📸 Screenshots (Optional)

> Add a screenshot of your Streamlit app or trend decomposition plot here.

- `model_training.py`: Trains Prophet model
- `app.py`: Streamlit app to visualize forecast
