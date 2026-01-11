import os, time, requests
import numpy as np, pandas as pd

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay, mean_absolute_error

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator

KLINES = "https://api.binance.us/api/v3/klines"
PRICE  = "https://api.binance.us/api/v3/ticker/price"

# -----------------------------
# Formatting helpers
# -----------------------------
def money(x: float) -> str:
    return f"{x:,.0f}"

def pct_prob(p: float) -> str:
    # 0.49 -> "49.00%"
    return f"{p * 100:.2f}%"

def pct_ret(r: float) -> str:
    # 0.0037 -> "+0.37%"
    sign = "+" if r >= 0 else ""
    return f"{sign}{r * 100:.2f}%"

def get_data(symbol="BTCUSDT", interval="1d", start="2024-01-01", cache="cache.csv", sleep=0.35):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    old = None
    if os.path.exists(cache):
        old = pd.read_csv(cache)
        old["open_time"]  = pd.to_datetime(old["open_time"], utc=True, errors="coerce")
        old["close_time"] = pd.to_datetime(old["close_time"], utc=True, errors="coerce")
        for c in ["open","high","low","close","volume"]:
            old[c] = pd.to_numeric(old[c], errors="coerce")
        old = old.dropna(subset=["open_time","close_time","open","high","low","close","volume"]).sort_values("open_time")
        if len(old):
            start_ms = int(old["open_time"].iloc[-1].value // 10**6) + 1

    rows, cur = [], start_ms
    while True:
        r = requests.get(KLINES, params={"symbol":symbol,"interval":interval,"limit":1000,"startTime":cur}, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows += data
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(sleep)

    cols = ["open_time","open","high","low","close","volume","close_time","qav","n","tbv","tqv","ignore"]
    new = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    if len(new):
        new["open_time"]  = pd.to_datetime(new["open_time"], unit="ms", utc=True)
        new["close_time"] = pd.to_datetime(new["close_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            new[c] = pd.to_numeric(new[c], errors="coerce")
        new = new.dropna(subset=["open_time","close_time","open","high","low","close","volume"]).sort_values("open_time")

    df = pd.concat([old, new], ignore_index=True) if old is not None else new
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    # completed candles only
    df = df[df["close_time"] <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)

    df.to_csv(cache, index=False)
    return df

def rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def make_features(df, rsi_period=14, ma=(5,10,20)):
    x = df.copy()
    x["ret1"] = x["close"].pct_change()
    x["oc"]   = (x["close"]-x["open"]) / x["open"]
    x["hl"]   = (x["high"]-x["low"]) / x["open"]
    x["vchg"] = x["volume"].pct_change()
    for w in ma:
        m = x["close"].rolling(w).mean()
        x[f"c_ma{w}"] = x["close"]/m - 1
    x[f"rsi{rsi_period}"] = rsi(x["close"], rsi_period)
    x["close_next"] = x["close"].shift(-1)
    x["y_cls"] = (x["close_next"] > x["close"]).astype(int)
    x["y_ret"] = x["close_next"]/x["close"] - 1
    return x.dropna().reset_index(drop=True)

def spot(symbol="BTCUSDT"):
    return float(requests.get(PRICE, params={"symbol":symbol}, timeout=15).json()["price"])

def main():
    # -----------------------------
    # 1) Data + features
    # -----------------------------
    df = get_data()
    feat = make_features(df)

    cols = ["ret1","oc","hl","vchg","c_ma5","c_ma10","c_ma20","rsi14"]
    n = len(feat); cut = int(n*0.8)
    train, test = feat.iloc[:cut], feat.iloc[cut:]

    Xtr, ytr = train[cols].to_numpy(np.float32), train["y_cls"].to_numpy(np.int32)
    Xte, yte = test[cols].to_numpy(np.float32),  test["y_cls"].to_numpy(np.int32)

    # -----------------------------
    # 2) Classifier (SVM)
    # -----------------------------
    svm = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])
    svm.fit(Xtr, ytr)

    pred = svm.predict(Xte)
    prob = svm.predict_proba(Xte)[:,1]

    print("Model: Support Vector Machine with Radial Basis Function kernel")
    print("Accuracy:", round(accuracy_score(yte, pred),4))
    print("F1 Score:", round(f1_score(yte, pred),4))
    print("Confusion Matrix:\n", confusion_matrix(yte, pred))
    print("Receiver Operating Characteristic Area Under the Curve:", round(roc_auc_score(yte, prob),4))

    # -----------------------------
    # 3) Regressor (Return magnitude)
    # -----------------------------
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(Xtr, train["y_ret"].to_numpy(np.float32))
    mae = mean_absolute_error(test["y_ret"].to_numpy(np.float32), rf.predict(Xte))
    print("Mean Absolute Error of next-day return:", round(mae,6))

    # -----------------------------
    # 4) Predict tomorrow
    # -----------------------------
    latestX = feat.iloc[-1][cols].to_numpy(np.float32).reshape(1,-1)
    p_up = float(svm.predict_proba(latestX)[0,1])
    p_down = 1 - p_up
    pred_ret = float(rf.predict(latestX)[0])

    today = spot()
    tomorrow = today*(1+pred_ret)

    print("\n" + "="*70)
    print("FINAL PREDICTION")
    print("="*70)
    print(f"Today spot price                                : {money(today)}")
    print(f"Predicted probability of upward movement tomorrow: {pct_prob(p_up)}")
    print(f"Predicted probability of downward movement tomorrow: {pct_prob(p_down)}")
    print(f"Predicted next-day return                        : {pct_ret(pred_ret)}")
    print(f"Predicted tomorrow price                         : {money(tomorrow)}")
    print("="*70 + "\n")

    # -----------------------------
    # 5) ROC curve plot
    # -----------------------------
    RocCurveDisplay.from_predictions(yte, prob, name="SVM (RBF)")
    plt.title("ROC Curve — Next-day Up/Down (Test)")
    plt.grid(True)
    plt.show()

    # -----------------------------
    # 6) Long-term trend plot (full range)
    # -----------------------------
    long_df = df.copy()
    long_df["ma20"] = long_df["close"].rolling(20).mean()
    long_df["ma60"] = long_df["close"].rolling(60).mean()

    plt.figure(figsize=(14, 6))

    # Price lines
    plt.plot(long_df["open_time"], long_df["close"], label="Close Price", linewidth=1.5)
    plt.plot(long_df["open_time"], long_df["ma20"], label="Moving Average 20 Days", linewidth=1.2)
    plt.plot(long_df["open_time"], long_df["ma60"], label="Moving Average 60 Days", linewidth=1.2)

    # ----- X axis: time (monthly ticks) -----
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))   # every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.xticks(rotation=45, ha="right")

    # ----- Y axis: price (more granular) -----
    ax.yaxis.set_major_locator(MultipleLocator(5000))  # every $5,000
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Labels and title
    plt.title("Long-term Close Price Trend (Full Range) [Binance US]", fontsize=14)
    plt.xlabel("Date (UTC)")
    plt.ylabel("Price (USDT)")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
