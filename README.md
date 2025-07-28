# 📈 Bitcoin Price Prediction using Machine Learning

## 📌 Objective

Develop a machine learning model to predict Bitcoin price trends based on historical data, helping traders and analysts make informed investment decisions.

## 🗃️ Dataset

- **Name**: Bitcoin Historical Price Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets), [Yahoo Finance](https://finance.yahoo.com), [CoinGecko API](https://www.coingecko.com/en/api)
- **Features**: Date, Open, High, Low, Close, Volume, etc.


## 🎯 Project Goals

### 1. Importing Libraries & Dataset
- Load essential Python libraries:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `sklearn` – preprocessing, model building
  - `xgboost` – advanced ML model
- Load dataset and inspect structure

### 2. Data Preprocessing
- Handle missing values using forward-fill
- Convert `Date` column to `datetime` format and set as index
- Normalize numerical features using `MinMaxScaler`
- Create new technical indicators:
  - **SMA**, **EMA**, **Bollinger Bands**, **RSI**
- Split data into training (80%) and testing (20%) sets

### 3. Exploratory Data Analysis (EDA)
- Plot Bitcoin price trends over time
- Analyze volume and volatility impact
- Heatmap for feature correlation

### 4. Model Training & Comparison
- Train and compare models:
  - `Linear Regression`
  - `Random Forest`
  - `Support Vector Machine (SVM)`
  - `XGBoost`
  - *(Optional)* `LSTM` using Keras for deep learning
- Evaluate models using RMSE

### 5. Model Evaluation & Prediction
- Metrics: `MAE`, `MSE`, `RMSE`, `R² Score`
- Predict future Bitcoin trends using trained model
- *(Optional)* Use real-time data from APIs like CoinGecko or Binance for predictions


## 🧪 Tools & Libraries

| Tool/Library | Use |
|--------------|-----|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical calculations |
| `matplotlib/seaborn` | Data visualization |
| `scikit-learn` | Preprocessing & ML modeling |
| `xgboost` | Gradient Boosting Regressor |
| `MinMaxScaler` | Feature normalization |
| `Keras/TensorFlow` | *(Optional)* LSTM model |
| `CoinGecko/Binance API` | *(Optional)* Real-time price data |


## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**


## 🚀 Future Improvements

- Integrate **LSTM** or **Transformer** for time-series deep learning
- Deploy as a web app with **Streamlit** or **Flask**
- Automate real-time prediction using **Binance API**
- Add **backtesting module** for trading strategies


## 📁 Folder Structure

bitcoin-price-prediction/
│
├── data/
│ └── bitcoin_price.csv
├── notebooks/
│ └── bitcoin_price_prediction.ipynb
├── models/
│ └── trained_model.pkl
├── src/
│ └── preprocessing.py
│ └── model_training.py
├── README.md
└── requirements.txt

