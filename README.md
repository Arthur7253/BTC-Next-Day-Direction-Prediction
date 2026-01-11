# BTC Next-Day Direction Prediction

This project predicts the **next-day price direction and return of Bitcoin (BTC/USDT)** using daily historical data from **Binance US** and machine learning models.

The system combines:
- **Support Vector Machine (SVM)** for next-day **up/down classification**
- **Random Forest Regression** for **next-day return magnitude**
- **Technical indicators** (returns, moving averages, RSI)
- **Visualizations** (ROC curve and long-term price trend)

This project is intended for **educational purposes** and demonstrates a complete end-to-end data mining workflow.

---

## Project Structure

```
BTC Next-Day Direction Prediction/
│
├── BTC Next-Day Direction Prediction.py   # Main Python script
├── cache.csv                              # Cached historical price data (auto-generated)
└── README.md                              # Project documentation
```

---

## Requirements

- Python **3.9+** recommended
- Internet connection (to fetch Binance US data)

### Required Python packages
```bash
pip install pandas numpy requests scikit-learn matplotlib
```

---

## How to Run (Compile & Execute)

### Step 1: Navigate to the project directory

```bash
cd "BTC Next-Day Direction Prediction"
```

### Step 2: Execute the program

```bash
python "BTC Next-Day Direction Prediction.py"
```

---

## What the Program Does

1. **Downloads daily BTC/USDT data** from Binance US  
2. **Caches data locally** (`cache.csv`) to avoid repeated downloads  
3. **Generates technical features**:
   - Daily return
   - Open-to-close return
   - High–low range
   - Volume change
   - Moving average ratios (5, 10, 20 days)
   - Relative Strength Index (RSI)
4. **Splits data chronologically** (80% training / 20% testing)
5. **Trains machine learning models**:
   - SVM classifier (price up or down)
   - Random Forest regressor (next-day return)
6. **Evaluates model performance**:
   - Accuracy, F1 score
   - Confusion matrix
   - ROC AUC and ROC curve
7. **Predicts tomorrow’s market behavior**:
   - Probability of price going up/down
   - Expected next-day return
   - Estimated next-day price
8. **Displays visualizations**:
   - ROC curve
   - Long-term price trend with moving averages

---

## Example Output

```
FINAL PREDICTION
======================================================================
Today spot price                                : 87,621
Predicted probability of upward movement tomorrow: 49.00%
Predicted probability of downward movement tomorrow: 51.00%
Predicted next-day return                        : +0.37%
Predicted tomorrow price                         : 87,949
======================================================================
```

---

## Plots Generated

- **ROC Curve**: Evaluates classification performance for predicting up/down days
- **Long-term Trend Plot**:
  - BTC closing price over time
  - 20-day and 60-day moving averages
  - Monthly time axis and fine-grained price scale

---

## Important Notes

- Only **completed daily candles (UTC)** are used to avoid look-ahead bias.
- Cryptocurrency markets are **highly volatile**; predictive accuracy is limited.
- Results should **not** be interpreted as financial or investment advice.

---

## Course Information

- **Project Name:** BTC Next-Day Direction Prediction
- **Language:** Python
- **Course:** CS425 – Data Mining
