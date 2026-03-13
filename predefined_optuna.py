"""
nextgen_hybrid_prophet_xgb_forecast_v5.py

Enhanced Prophet + XGBoost residual correction hybrid forecaster for BTC-USD

Includes:
- Robust Yahoo Finance download
- Automatic lag selection via PACF
- PACF visualization (diagnostic)
- Short-term trend features (EMA, SMA)
- Predefined XGBoost hyperparameters
- Hybrid forecast with uncertainty
- Directional accuracy + statistical significance testing
- Learning curve visualization (directional accuracy)
- Extended holdout evaluation
- Short-horizon (10-day) + extended forecast tables

Educational / research use only — not financial advice.
"""

# =========================
# Imports
# =========================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import accuracy_score
import xgboost as xgb
from datetime import timedelta
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import binomtest

# =========================
# Config
# =========================
TICKER = "BTC-USD"
START = "2021-01-01"
HORIZON_DAYS = 365
TEST_LAST_DAYS = 90
RANDOM_STATE = 42
MAX_LAG = 30

best_params_fixed = {
    "max_depth": 3,
    "learning_rate": 0.010349570637285655,
    "subsample": 0.8021272578985711,
    "colsample_bytree": 0.7728798862419759,
    "objective": "reg:squarederror",
    "seed": RANDOM_STATE,
    "verbosity": 0,
}

# =========================
# 1) Download data
# =========================
print("📥 Downloading data from Yahoo Finance...")
df_raw = yf.download(TICKER, start=START, progress=False)

if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = ["_".join(col).strip() for col in df_raw.columns]

price_candidates = [c for c in df_raw.columns if "close" in c.lower()]
volume_candidates = [c for c in df_raw.columns if "volume" in c.lower()]

price_col = price_candidates[0]
volume_col = volume_candidates[0] if volume_candidates else None

df_raw = df_raw[[price_col, volume_col]].rename(
    columns={price_col: "close", volume_col: "volume"}
)
df_raw.index = pd.to_datetime(df_raw.index)
df_raw = df_raw.sort_index()

print(
    f"✅ Using price column: {price_col}, volume column: {volume_col}, {len(df_raw)} rows"
)

# =========================
# 2) Feature engineering
# =========================
df = df_raw.copy()
df["return"] = df["close"].pct_change()
df["log_close"] = np.log(df["close"])

# -------------------------
# PACF lag selection
# -------------------------
pacf_vals = pacf(df["return"].dropna(), nlags=MAX_LAG)
N = len(df)
significant_lags = [
    i for i, v in enumerate(pacf_vals)
    if i > 0 and abs(v) > 2 / np.sqrt(N)
]

for l in significant_lags:
    df[f"lag_ret_{l}"] = df["return"].shift(l)

# Rolling stats
for w in [7, 14, 30]:
    df[f"roll_mean_ret_{w}"] = df["return"].rolling(w).mean()
    df[f"roll_std_ret_{w}"] = df["return"].rolling(w).std()
    df[f"roll_vol_{w}"] = df["volume"].rolling(w).mean()

# EMA / SMA
for span in [7, 14, 30]:
    df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

for w in [7, 14, 30]:
    df[f"sma_{w}"] = df["close"].rolling(w).mean()

df = df.dropna().copy()

# -------------------------
# PACF visualization (ADDED)
# -------------------------
fig, ax = plt.subplots(figsize=(8, 4))
plot_pacf(df["return"], lags=MAX_LAG, method="ywm", zero=False, ax=ax)
ax.set_title("PACF of BTC Returns")
ax.set_xlabel("Lag")
ax.set_ylabel("Partial Autocorrelation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 3) Prophet baseline
# =========================
print("⚙️ Fitting Prophet baseline...")
prophet_df = pd.DataFrame({"ds": df.index, "y": df["log_close"].values})
m = Prophet(
    interval_width=0.95,
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
)
m.fit(prophet_df)

future_prophet = m.make_future_dataframe(periods=HORIZON_DAYS)
forecast_prophet = m.predict(future_prophet)

fc = forecast_prophet[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")

prophet_in_sample = fc.reindex(df.index).dropna()
df = df.loc[prophet_in_sample.index]
df["prophet_log"] = prophet_in_sample["yhat"].values
df["residual"] = df["log_close"] - df["prophet_log"]

# =========================
# 4) XGBoost residuals
# =========================
print("⚙️ Preparing XGBoost features...")
feature_cols = [
    c for c in df.columns
    if isinstance(c, str) and c.startswith(("lag_", "roll_", "ema_", "sma_"))
]

X = df[feature_cols]
y_res = df["residual"]

split_idx = -TEST_LAST_DAYS
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_res.iloc[:split_idx], y_res.iloc[split_idx:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

bst = xgb.train(best_params_fixed, dtrain, num_boost_round=200, verbose_eval=False)
pred_res_test = bst.predict(dtest)

mse_res = ((y_test - pred_res_test) ** 2).mean()
print(f"✅ XGBoost residual model trained. Residual test MSE: {mse_res:.6e}")

# =========================
# 5) Hybrid forecast
# =========================
print(f"🔮 Producing combined {HORIZON_DAYS}-day forecast...")

prophet_future = fc.reindex(
    pd.date_range(
        start=df.index.max() + timedelta(days=1),
        periods=HORIZON_DAYS,
        freq="D",
    )
).ffill()

prophet_future = prophet_future.rename(columns={"yhat": "prophet_log"})

history = df.copy()
future_rows = []

for _ in range(HORIZON_DAYS):
    next_date = history.index.max() + timedelta(days=1)
    row = {}

    for col in feature_cols:
        if col.startswith("lag_ret_"):
            l = int(col.split("_")[-1])
            row[col] = history["return"].iloc[-l]
        elif col.startswith("roll_mean"):
            w = int(col.split("_")[-1])
            row[col] = history["return"].iloc[-w:].mean()
        elif col.startswith("roll_std"):
            w = int(col.split("_")[-1])
            row[col] = history["return"].iloc[-w:].std()
        elif col.startswith("roll_vol"):
            w = int(col.split("_")[-1])
            row[col] = history["volume"].iloc[-w:].mean()
        elif col.startswith("ema_"):
            span = int(col.split("_")[-1])
            row[col] = history["close"].iloc[-span:].ewm(span=span).mean().iloc[-1]
        elif col.startswith("sma_"):
            w = int(col.split("_")[-1])
            row[col] = history["close"].iloc[-w:].mean()

    future_rows.append(pd.Series(row, name=next_date))

    history.loc[next_date] = history.iloc[-1]

X_future = pd.DataFrame(future_rows).ffill()
pred_res_future = bst.predict(xgb.DMatrix(X_future))

combined_log = prophet_future["prophet_log"].values[:HORIZON_DAYS] + pred_res_future
combined_price = np.exp(combined_log)

resid_std = y_train.std()
z = 1.96

combined_low = combined_price * np.exp(-z * resid_std)
combined_high = combined_price * np.exp(z * resid_std)

result_df = pd.DataFrame(
    {
        "combined_price": combined_price,
        "combined_low": combined_low,
        "combined_high": combined_high,
    },
    index=prophet_future.index[:HORIZON_DAYS],
)

# =========================
# 6) Directional accuracy + p-value
# =========================
prophet_test_log = fc.reindex(X_test.index)["yhat"].values
pred_log_test = prophet_test_log + pred_res_test

pred_price_test = np.exp(pred_log_test)
true_price_test = df.loc[X_test.index, "close"].values

pred_dir = np.diff(pred_price_test) > 0
true_dir = np.diff(true_price_test) > 0

dir_acc = accuracy_score(true_dir, pred_dir)
p_value = binomtest(
    k=(pred_dir == true_dir).sum(),
    n=len(pred_dir),
    p=0.5,
    alternative='greater'
).pvalue

print(f"Directional accuracy on holdout: {dir_acc:.4f}")
print("p-value vs random chance:", format(p_value, ".6f"))

# =========================
# 7) Learning curve
# =========================
print("⚙️ Computing learning curve...")
train_sizes = np.linspace(0.1, 1.0, 5)
train_acc, test_acc = [], []

prophet_train_log = fc.reindex(X_train.index)["yhat"].values
prophet_test_log = fc.reindex(X_test.index)["yhat"].values  # FIXED: renamed from prophet_test_log_full

for frac in train_sizes:
    n = int(frac * len(X_train))

    dtrain_part = xgb.DMatrix(X_train.iloc[:n], label=y_train.iloc[:n])
    bst_part = xgb.train(best_params_fixed, dtrain_part, num_boost_round=200, verbose_eval=False)

    pred_train = np.exp(prophet_train_log[:n] + bst_part.predict(dtrain_part))
    true_train = df.loc[X_train.index[:n], "close"].values

    train_acc.append(
        accuracy_score(np.diff(true_train) > 0, np.diff(pred_train) > 0)
    )

    pred_test = np.exp(
        prophet_test_log + bst_part.predict(xgb.DMatrix(X_test))  # FIXED: uses prophet_test_log
    )
    test_acc.append(
        accuracy_score(np.diff(true_price_test) > 0, np.diff(pred_test) > 0)
    )

plt.figure(figsize=(8, 5))
plt.plot(train_sizes * 100, train_acc, marker="o", label="Train Accuracy")
plt.plot(train_sizes * 100, test_acc, marker="o", label="Test Accuracy")
plt.axhline(0.5, linestyle="--", color="gray", label="Random chance")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Directional Accuracy")
plt.title("Learning Curve: Directional Accuracy vs Training Size")
plt.ylim([0.45, max(max(train_acc), max(test_acc)) + 0.05])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 8) Forecast plot
# =========================
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["close"], label="Historical")
plt.plot(result_df.index, result_df["combined_price"], label="Hybrid forecast")
plt.fill_between(
    result_df.index,
    result_df["combined_low"],
    result_df["combined_high"],
    alpha=0.15,
    label="Confidence interval",
)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("BTC-USD Hybrid Prophet + XGBoost Forecast")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 9) Output
# =========================
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("\n🔮 Combined forecast (first 10 days):")
print(result_df.head(10))

print("\n🔮 Combined forecast (first 15 days):")
print(result_df.head(15))

result_df.to_csv("nextgen_hybrid_forecast_results.csv")
print("\n✅ Forecast saved to nextgen_hybrid_forecast_results.csv")



