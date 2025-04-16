**Stock Market Prediction using ML & ANN (MSFT & S&P 500)**
-----------------------------------------------------------

**Course Instructor:**
---------------------

Dr. Kavita Khanna

**Contributors:**
----------------

**Student-1**

Name: Prashant Singh

Enrollment No.: SAU/CS/Mtech(CS)/2024/04

Program: Mtech(CS)

**Student-2**

Name: Sakshi Wagh

Enrollment No.: SAU/CS/Mtech(CS)/2024/06

Program: Mtech(CS)


This project implements multiple machine learning (ML) and deep learning (DL) models to predict stock price movements using historical data. It includes experiments on both individual stock data (MSFT) and index data (S&P 500) using yfinance. The goal is to predict whether the stock/index will close higher the next day and evaluate the precision of various models in real-time backtesting scenarios.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 **Data Fetching and Preprocessing**

✅ MSFT (Microsoft)

Used yfinance to download the full historical data from 1986 to 2025 (includes simulated future data).

Removed irrelevant columns and applied a RobustScaler for normalization.

Created a binary target:

1: Next day’s close > Today’s close

0: Otherwise

Used previous day's scaled values of key predictors: ["Close", "High", "Low", "Open", "Volume"].

✅ S&P 500 Index

Used yfinance to download data for ^GSPC.

Cached the data locally (sp500.csv) to avoid repeated downloads.

Removed "Dividends" and "Stock Splits" columns.

Applied the same binary target transformation.

Used predictors: ["Close", "Volume", "Open", "High", "Low"].

Filtered data from 1990 onwards for consistency.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 **Models Implemented**

**1. Random Forest Classifier**
 
Used sklearn.ensemble.RandomForestClassifier.

Tuned with:

n_estimators=100

min_samples_split=100 (S&P) or 200 (MSFT)

Evaluated on:

Static test set (last 100 entries)

Backtesting using a sliding window approach

**2. Logistic Regression**
 
Used sklearn.linear_model.LogisticRegression.

Provides a simpler, interpretable baseline.

Evaluated similarly as Random Forest.

**3. Artificial Neural Network (ANN)**

For MSFT:

Built using Keras Sequential with:

Layers: 128 → 64 → 32 → 1 (sigmoid)

Activation: LeakyReLU with LayerNormalization

Optimizer: Adam (lr=0.001)

Loss: binary_crossentropy

Metrics: Precision

Epochs: 100

For S&P 500:
Deeper architecture with regularization:

Layers: 128 → 64 → 32 → 1

BatchNormalization and Dropout added

Optimizer: Adam (lr=0.0001)

Trained with early stopping support

### 🔍 **Model Comparison Summary**

| Model               | Dataset Used         | Preprocessing                         | Feature Scaling     | Architecture / Parameters                                                                 | Backtesting Applied | Final Precision (Backtested) |
|--------------------|----------------------|---------------------------------------|---------------------|--------------------------------------------------------------------------------------------|----------------------|------------------------------|
| Random Forest       | MSFT                 | Binary target, shifted predictors     | ✅ RobustScaler      | n_estimators=100, min_samples_split=200                                                    | ✅ Yes               | 0.51                         |
| Logistic Regression | MSFT                 | Binary target, shifted predictors     | ✅ RobustScaler      | Default                                                                                     | ✅ Yes               | 0.51                         |
| ANN (Keras)         | MSFT                 | Binary target, shifted predictors     | ✅ RobustScaler      | Dense(128→64→32→1), LeakyReLU, LayerNorm, BinaryCrossentropy, Adam                         | ✅ Yes               | 0.52                         |
| Random Forest       | S&P 500 (^GSPC)      | Binary target (Tomorrow > Today)      | ❌ Not Scaled        | n_estimators=100, min_samples_split=100                                                    | ✅ Yes               | 0.52                         |
| Logistic Regression | S&P 500 (^GSPC)      | Binary target (Tomorrow > Today)      | ❌ Not Scaled        | Default                                                                                     | ✅ Yes               | 0.52                         |
| ANN (Keras)         | S&P 500 (^GSPC)      | Binary target (Tomorrow > Today)      | ❌ Not Scaled        | Dense(128→64→32→1), LeakyReLU, BatchNorm, Dropout, BinaryCrossentropy, Adam               | ✅ Yes               | 0.53                         |

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔁**Backtesting Framework**

Implemented custom backtesting functions to mimic real-world rolling predictions.

🔄 **Logic:**

Train model on rolling windows

Predict next step days

Append predictions and compute performance

✅ **Functions:**
backtest() → for Random Forest & Logistic Regression

backtestANN() / backtest_ann() → for ANN

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📈 **Evaluation Metric:**

Precision Score: Measures the proportion of predicted positive (upward) movements that were correct.

### 📊 **Results Summary**

| Model               | Dataset   | Precision (Static) | Precision (Backtesting) |
|--------------------|-----------|---------------------|--------------------------|
| Random Forest       | MSFT      | 0.48                | 0.51                     |
| Logistic Regression | MSFT      | 0.51                | 0.51                     |
| ANN                 | MSFT      | 0.52                | 0.52                     |
| Random Forest       | S&P 500   | 0.45                | 0.51                     |
| Logistic Regression | S&P 500   | 0.45                | 0.51                     |
| ANN                 | S&P 500   | 0.53                | 0.52                     |

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**ANN models slightly outperform others in precision across both datasets.**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 **Key Takeaways**

ANN models with appropriate normalization and architecture yield slightly better precision than traditional ML models.

Backtesting with walk-forward validation provides a more reliable assessment than static evaluation.

This framework is extendable to other stocks and indices with minimal changes.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧪 **Future Work**

Integrate real-time data updates with scheduled predictions.

Add more sophisticated features like technical indicators (RSI, MACD).

Use of advanced DL models (LSTM, Transformer) for time-series analysis.

Experiment with multi-stock joint modeling or portfolio optimization.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**This project is part of an academic and applied machine learning initiative. Built using Python, scikit-learn, TensorFlow, and yfinance.**






