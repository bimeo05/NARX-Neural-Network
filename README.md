# Predicting Stock Returns with NARX Neural Networks

## 📝 Project Overview

This research enhances the traditional event study methodology by applying Artificial Intelligence to predict normal returns of green and brown portfolios. Instead of relying on standard Capital Asset Pricing Model (CAPM) or market model approaches, this project implements a **Nonlinear AutoRegressive with eXogenous inputs (NARX) neural network** to improve prediction accuracy.

The study specifically examines market reactions to the European Green Deal's ambitions leaked on November 29, 2019, focusing on how this climate policy affected green versus brown companies in the Eurozone.

## 🔍 Methodology

### Research Framework

1. **Data Preparation**
   - Define input and output variables
   - Data acquisition
   - Preprocessing
   - Normalization
   - Structuring data

2. **Algorithm Definition**
   - Choose predictor
   - Configure architecture

3. **Training**
   - Define algorithm
   - Adjust parameters
   - Perform training

4. **Forecasting Evaluation**
   - Define metrics
   - Evaluate accuracy

## 💾 Data Preparation

### Portfolio Construction

- **Data Source**: Stocks from EURO STOXX 50 index (50 largest and most liquid blue-chip companies in the Eurozone)
- **Time Period**: Daily data from September 13, 2018 to December 20, 2019
- **Green vs. Brown Classification**: Based on Environmental Pillar Score from Refinitiv Eikon database
  - **Green companies**: Rating of A or A+
  - **Brown companies**: All other ratings

### Input Variables

The neural network model uses the following input variables:
- Portfolio historical returns (with lags)
- Cyclical features (day of month, day of week, etc.)
- EURO STOXX 50 index returns (market proxy)
- EURO STOXX 50 Volatility index
- Macroeconomic variables (1-week Euribor rate)

## 🧮 Data Preprocessing

### Stationarity in Time Series

Stock prices are converted to log returns to ensure stationarity:
```
r(i,t) = ln(p(i,t)) - ln(p(i,t-1))
```
Where:
- r(i,t) = logarithmic return of stock i for day t
- p(i,t) = adjusted close price of stock i for day t

### Data Transformation

- **Standardization**: Used for financial variables to normalize scale (mean of 0, standard deviation of 1)
- **Cyclical Variables**: Transformed using sine and cosine functions to preserve cyclical nature
- **Missing Data**: Forward-filled with last observed value

## 🧠 Neural Network Implementation

### NARX Neural Network

The NARX model predicts the next value of portfolio returns based on:
- Previous values of portfolio returns (autoregressive component)
- Previous values of independent variables (exogenous inputs)

Model equation:
```
y(t) = f(y(t-1), y(t-2), ..., y(t-n), ..., x(t-1), x(t-2), ..., x(t-n))
```

### Model Training

- **Platform**: MATLAB R2024b with neural network builder
- **Data Split**: 70% training, 15% validation, 15% testing
- **Training Algorithm**: Levenberg-Marquardt (optimizing for MSE)
- **Early Stopping**: To prevent overfitting

## 📊 Performance Evaluation

### Metrics

- **Root Mean Square Error (RMSE)**: For interpretability (same unit as target variable)
- **R-value**: To measure relationship strength between model output and actual data

### Comparison with Traditional Event Study

Traditional CAPM approach:
```
R(k,t) - r(f,t) = a(k) + β(k)(r(m,t) - r(f,t)) + ε(k,t)
```

The study compares abnormal returns (AR) and cumulative abnormal returns (CAR) calculated using:
1. Traditional CAPM-based expected returns
2. NARX neural network predicted expected returns

## 🔬 Statistical Analysis

Abnormal returns are calculated as:
```
AR(k,t) = R(k,t) - E[R(k,t)]
```

Cumulative abnormal returns are calculated as:
```
CAR(k)[t1,t2] = Σ(t=t1 to t2) AR(k,t)
```

Statistical significance testing uses standard T-tests.

## 🛠️ Implementation Details

- **Code**: MATLAB R2024b implementation of NARX neural network
- **Libraries**: Built-in MATLAB Neural Network Toolbox
- **Settings**: 10-neuron hidden layer, time delay of 2 (default configuration)

## 📚 References

- Armitage, S. (1995). Event study methods and evidence on their performance.
- Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and long-short term memory.
- Borghesi, R., Houston, J. F., & Naranjo, A. (2022). Corporate socially responsible investments: CEO altruism, reputation, and shareholder interests.
- Cavalcante, R. C., Brasileiro, R. C., Souza, V. L., Nobrega, J. P., & Oliveira, A. L. (2016). Computational intelligence and financial markets: A survey and future directions.
- Mueller, C., et al. (2023). [Paper on European Green Deal effects]
- Palit, A. K., & Popovic, D. (2006). Computational intelligence in time series forecasting: Theory and engineering applications.
