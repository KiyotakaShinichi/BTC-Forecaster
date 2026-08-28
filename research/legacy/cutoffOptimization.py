"""
nextgen_hybrid_prophet_xgb_montecarlo_WITH_AUTO_CUTOFF.py

Enhanced BTC-USD hybrid forecaster with:
- AUTOMATIC OPTIMAL CUTOFF SELECTION (NEW!)
- Automatic PACF lag selection
- Automatic rolling windows, EMA, SMA selection
- Monte Carlo simulation for long-term probabilistic forecasts
- Prophet baseline + XGBoost residual correction
- Directional accuracy + p-value testing
- Learning curve visualization
- GARCH volatility modeling

NEW FEATURE:
Instead of hardcoding START = "2021-01-01", this script:
1. Tests multiple cutoff dates
2. Evaluates each on directional accuracy + MAE
3. Automatically selects the OPTIMAL cutoff
4. Uses that for the final forecast

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
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
from datetime import timedelta, datetime
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import binomtest
from arch import arch_model

# =========================
# Config
# =========================
TICKER = "BTC-USD"
# START will be determined automatically!
HORIZON_DAYS = 90
TEST_LAST_DAYS = 30
RANDOM_STATE = 42
MAX_LAG = 60
ROLL_WINDOWS_CANDIDATES = [3, 5, 7, 14, 21, 30, 60]
EMA_SMA_CANDIDATES = [3, 5, 7, 14, 21, 30]

xgb_params = {
    "max_depth": 3,
    "learning_rate": 0.010349570637285655,
    "subsample": 0.8021272578985711,
    "colsample_bytree": 0.7728798862419759,
    "objective": "reg:squarederror",
    "seed": RANDOM_STATE,
    "verbosity": 0,
}

MONTE_CARLO_RUNS = 1000

# Cutoff candidates to test
CUTOFF_CANDIDATES = [
    "2015-01-01",
    "2017-01-01",
    "2018-01-01",
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
    "2022-01-01",
]

print("=" * 80)
print("🚀 BTC HYBRID FORECASTER WITH AUTOMATIC OPTIMAL CUTOFF")
print("=" * 80)

# =========================
# STEP 1: FIND OPTIMAL CUTOFF
# =========================
print("\n" + "=" * 80)
print("STEP 1: FINDING OPTIMAL TRAINING CUTOFF")
print("=" * 80)

print(f"\n📥 Downloading full BTC history...")
df_full = yf.download(TICKER, start="2015-01-01", progress=False)

if isinstance(df_full.columns, pd.MultiIndex):
    df_full.columns = ["_".join(col).strip() for col in df_full.columns]

price_col = [c for c in df_full.columns if "close" in c.lower()][0]
volume_col = [c for c in df_full.columns if "volume" in c.lower()][0]

df_full = df_full[[price_col, volume_col]].rename(
    columns={price_col: "close", volume_col: "volume"}
)
df_full.index = pd.to_datetime(df_full.index)
df_full = df_full.sort_index()

print(f"✅ Downloaded {len(df_full)} rows ({df_full.index.min().date()} → {df_full.index.max().date()})")


def quick_evaluate_cutoff(df_raw, cutoff_date, test_days=90):
    """Quick evaluation of a cutoff date - returns directional accuracy"""

    # Basic feature engineering
    df = df_raw.copy()
    df["return"] = df["close"].pct_change()
    df["log_close"] = np.log(df["close"])

    # Simple features (fixed for speed)
    for l in [1, 7, 14]:
        df[f"lag_ret_{l}"] = df["return"].shift(l)

    for w in [7, 14, 30]:
        df[f"roll_mean_ret_{w}"] = df["return"].rolling(w).mean()
        df[f"roll_std_ret_{w}"] = df["return"].rolling(w).std()

    df = df.dropna().copy()

    # Prophet
    prophet_df = pd.DataFrame({"ds": df.index, "y": df["log_close"].values})
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    m.fit(prophet_df)

    forecast_prophet = m.predict(m.make_future_dataframe(periods=0))
    fc = forecast_prophet[["ds", "yhat"]].set_index("ds")

    prophet_in_sample = fc.reindex(df.index).dropna()
    df = df.loc[prophet_in_sample.index]
    df["prophet_log"] = prophet_in_sample["yhat"].values
    df["residual"] = df["log_close"] - df["prophet_log"]

    # XGBoost
    feature_cols = [c for c in df.columns if c.startswith(("lag_", "roll_"))]
    X = df[feature_cols]
    y_res = df["residual"]

    split_idx = -test_days
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_res.iloc[:split_idx], y_res.iloc[split_idx:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(xgb_params, dtrain, num_boost_round=100, verbose_eval=False)
    pred_res_test = bst.predict(dtest)

    # Evaluate
    pred_log_test = fc.reindex(X_test.index)["yhat"].values + pred_res_test
    pred_price_test = np.exp(pred_log_test)
    true_price_test = df.loc[X_test.index, "close"].values

    pred_dir = np.diff(pred_price_test) > 0
    true_dir = np.diff(true_price_test) > 0
    dir_acc = accuracy_score(true_dir, pred_dir)
    mae = mean_absolute_error(true_price_test, pred_price_test)

    return {
        "dir_acc": dir_acc,
        "mae": mae,
        "training_days": len(df),
    }


print(f"\n🔄 Testing {len(CUTOFF_CANDIDATES)} cutoff dates...")
print(f"{'Cutoff':<12} {'Days':<8} {'Dir Acc':<12} {'MAE':<15}")
print("-" * 80)

results = []
for cutoff in CUTOFF_CANDIDATES:
    df_subset = df_full[df_full.index >= cutoff].copy()

    if len(df_subset) < 365:
        print(f"{cutoff:<12} {'SKIP':<8} {'Insufficient data':<12}")
        continue

    try:
        metrics = quick_evaluate_cutoff(df_subset, cutoff, test_days=90)
        results.append({
            "cutoff": cutoff,
            "training_days": metrics["training_days"],
            "dir_acc": metrics["dir_acc"],
            "mae": metrics["mae"],
        })
        print(f"{cutoff:<12} {metrics['training_days']:<8} "
              f"{metrics['dir_acc']:<12.4f} ${metrics['mae']:<14,.0f}")
    except Exception as e:
        print(f"{cutoff:<12} {'ERROR':<8} {str(e)[:30]}")

if len(results) == 0:
    print("❌ No valid cutoffs found, using default 2021-01-01")
    OPTIMAL_START = "2021-01-01"
else:
    results_df = pd.DataFrame(results)

    # Score each cutoff (weighted)
    results_df["score"] = (
            results_df["dir_acc"] * 0.6 +  # 60% weight on accuracy
            (1 - results_df["mae"] / results_df["mae"].max()) * 0.2 +  # 20% on MAE
            (results_df["training_days"] / results_df["training_days"].max()) * 0.2  # 20% on data
    )

    best_idx = results_df["score"].idxmax()
    best_cutoff = results_df.loc[best_idx]

    OPTIMAL_START = best_cutoff["cutoff"]

    print("\n" + "=" * 80)
    print("🎯 OPTIMAL CUTOFF SELECTED")
    print("=" * 80)
    print(f"\n✅ Optimal cutoff: {OPTIMAL_START}")
    print(f"   Training days: {best_cutoff['training_days']}")
    print(f"   Directional accuracy: {best_cutoff['dir_acc']:.4f} ({best_cutoff['dir_acc'] * 100:.2f}%)")
    print(f"   MAE: ${best_cutoff['mae']:,.0f}")
    print(f"\n💡 This cutoff maximizes predictive accuracy while balancing data quantity")

# =========================
# STEP 2: FULL FORECAST WITH OPTIMAL CUTOFF
# =========================
print("\n" + "=" * 80)
print(f"STEP 2: FULL FORECAST USING OPTIMAL CUTOFF ({OPTIMAL_START})")
print("=" * 80)

print(f"\n📥 Downloading data from {OPTIMAL_START}...")
df_raw = yf.download(TICKER, start=OPTIMAL_START, progress=False)

if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = ["_".join(col).strip() for col in df_raw.columns]

price_col = [c for c in df_raw.columns if "close" in c.lower()][0]
volume_col = [c for c in df_raw.columns if "volume" in c.lower()][0]

df_raw = df_raw[[price_col, volume_col]].rename(columns={price_col: "close", volume_col: "volume"})
df_raw.index = pd.to_datetime(df_raw.index)
df_raw = df_raw.sort_index()

print(f"✅ Downloaded {len(df_raw)} rows ({df_raw.index.min().date()} → {df_raw.index.max().date()})")

# =========================
# 2) Feature engineering
# =========================
df = df_raw.copy()
df["return"] = df["close"].pct_change()
df["log_close"] = np.log(df["close"])

# --------- Automatic PACF lag selection ---------
print("🔍 Computing PACF for automatic lag selection...")
pacf_vals = pacf(df["return"].dropna(), nlags=MAX_LAG)
N = len(df)
significant_lags = [i for i, v in enumerate(pacf_vals) if i > 0 and abs(v) > 2 / np.sqrt(N)]
if not significant_lags:
    significant_lags = [1]

print(f"✅ Selected lags: {significant_lags}")

for l in significant_lags:
    df[f"lag_ret_{l}"] = df["return"].shift(l)

# --------- PACF visualization ---------
fig, ax = plt.subplots(figsize=(8, 4))
plot_pacf(df["return"].dropna(), lags=MAX_LAG, method="ywm", zero=False, ax=ax)
ax.axhline(2 / np.sqrt(N), linestyle="--", color="red", alpha=0.6)
ax.axhline(-2 / np.sqrt(N), linestyle="--", color="red", alpha=0.6)
ax.set_title(f"PACF of BTC Returns (Cutoff: {OPTIMAL_START})")
ax.set_xlabel("Lag")
ax.set_ylabel("Partial Autocorrelation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pacf_diagnostic.png", dpi=150, bbox_inches="tight")
print("📊 PACF plot saved")
plt.show()

# --------- Automatic rolling windows selection ---------
print("🔍 Selecting optimal rolling windows...")
rolling_features = {}
for w in ROLL_WINDOWS_CANDIDATES:
    roll_mean = df["return"].rolling(w).mean()
    roll_std = df["return"].rolling(w).std()
    roll_vol = df["volume"].rolling(w).mean()
    corr_mean = abs(roll_mean.corr(df["return"].shift(-1)))
    corr_std = abs(roll_std.corr(df["return"].shift(-1)))
    corr_vol = abs(roll_vol.corr(df["return"].shift(-1)))
    rolling_features[w] = (corr_mean + corr_std + corr_vol) / 3

top_roll_windows = sorted(rolling_features, key=rolling_features.get, reverse=True)[:3]
print(f"✅ Selected rolling windows: {top_roll_windows}")

for w in top_roll_windows:
    df[f"roll_mean_ret_{w}"] = df["return"].rolling(w).mean()
    df[f"roll_std_ret_{w}"] = df["return"].rolling(w).std()
    df[f"roll_vol_{w}"] = df["volume"].rolling(w).mean()

# --------- Automatic EMA and SMA selection ---------
print("🔍 Selecting optimal EMA and SMA periods...")
ema_features = {}
sma_features = {}
for span in EMA_SMA_CANDIDATES:
    ema = df["close"].ewm(span=span, adjust=False).mean()
    sma = df["close"].rolling(span).mean()
    corr_ema = abs(ema.corr(df["return"].shift(-1)))
    corr_sma = abs(sma.corr(df["return"].shift(-1)))
    ema_features[span] = corr_ema
    sma_features[span] = corr_sma

top_ema = sorted(ema_features, key=ema_features.get, reverse=True)[:3]
top_sma = sorted(sma_features, key=sma_features.get, reverse=True)[:3]
print(f"✅ Selected EMA periods: {top_ema}")
print(f"✅ Selected SMA periods: {top_sma}")

for span in top_ema:
    df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
for span in top_sma:
    df[f"sma_{span}"] = df["close"].rolling(span).mean()

df = df.dropna().copy()

# =========================
# 3) Prophet baseline
# =========================
print("⚙️ Fitting Prophet baseline...")
prophet_df = pd.DataFrame({"ds": df.index, "y": df["log_close"].values})
m = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
m.fit(prophet_df)

future_prophet = m.make_future_dataframe(periods=HORIZON_DAYS)
forecast_prophet = m.predict(future_prophet)
fc = forecast_prophet[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")

prophet_in_sample = fc.reindex(df.index).dropna()
df = df.loc[prophet_in_sample.index]
df["prophet_log"] = prophet_in_sample["yhat"].values
df["residual"] = df["log_close"] - df["prophet_log"]

print(f"✅ Prophet fitted on {len(df)} samples")

# =========================
# 4) XGBoost residuals
# =========================
print("⚙️ Training XGBoost on residuals...")
feature_cols = [c for c in df.columns if c.startswith(("lag_", "roll_", "ema_", "sma_"))]
print(f"📊 Using {len(feature_cols)} features")

X = df[feature_cols]
y_res = df["residual"]

split_idx = -TEST_LAST_DAYS
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_res.iloc[:split_idx], y_res.iloc[split_idx:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

bst = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=False)
pred_res_test = bst.predict(dtest)

mse_res = ((y_test - pred_res_test) ** 2).mean()
print(f"✅ XGBoost trained. MSE: {mse_res:.6e}")

# =========================
# 4.5) GARCH volatility model
# =========================
print("⚙️ Fitting GARCH(1,1)...")
garch_resid = y_train - bst.predict(dtrain)
garch = arch_model(garch_resid * 100, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
garch_fitted = garch.fit(disp="off")
print("✅ GARCH fitted")

# =========================
# 5) Hybrid forecast + Monte Carlo
# =========================
print(f"🔮 Producing {HORIZON_DAYS}-day forecast...")

future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=HORIZON_DAYS, freq="D")
prophet_future_log = fc.reindex(future_dates)["yhat"].ffill().values

# Build future features
print("🔧 Constructing future features...")
history_df = df.copy()
future_features = []

for i, next_date in enumerate(future_dates):
    row = {}
    for col in feature_cols:
        if col.startswith("lag_ret_"):
            l = int(col.split("_")[-1])
            row[col] = history_df["return"].iloc[-l] if len(history_df) >= l else 0.0
        elif col.startswith("roll_mean_ret_"):
            w = int(col.split("_")[-1])
            row[col] = history_df["return"].iloc[-w:].mean() if len(history_df) >= w else history_df["return"].mean()
        elif col.startswith("roll_std_ret_"):
            w = int(col.split("_")[-1])
            row[col] = history_df["return"].iloc[-w:].std() if len(history_df) >= w else history_df["return"].std()
        elif col.startswith("roll_vol_"):
            w = int(col.split("_")[-1])
            row[col] = history_df["volume"].iloc[-w:].mean() if len(history_df) >= w else history_df["volume"].mean()
        elif col.startswith("ema_"):
            span = int(col.split("_")[-1])
            row[col] = history_df["close"].ewm(span=span, adjust=False).mean().iloc[-1] if len(history_df) >= span else \
            history_df["close"].mean()
        elif col.startswith("sma_"):
            w = int(col.split("_")[-1])
            row[col] = history_df["close"].iloc[-w:].mean() if len(history_df) >= w else history_df["close"].mean()

    future_features.append(row)
    if i < len(future_dates) - 1:
        dummy_row = history_df.iloc[-1].copy()
        history_df = pd.concat([history_df, pd.DataFrame([dummy_row], index=[next_date])])

X_future = pd.DataFrame(future_features, index=future_dates)[feature_cols]
pred_res_future = bst.predict(xgb.DMatrix(X_future))

combined_log = prophet_future_log + pred_res_future
combined_price = np.exp(combined_log)

print("✅ Point forecast generated")

# Monte Carlo simulation
print(f"🎲 Running Monte Carlo ({MONTE_CARLO_RUNS} runs)...")
garch_forecast = garch_fitted.forecast(horizon=HORIZON_DAYS)
cond_var = garch_forecast.variance.values[-1] / 10000
cond_std = np.sqrt(cond_var)

np.random.seed(RANDOM_STATE)
mc_matrix = np.zeros((MONTE_CARLO_RUNS, HORIZON_DAYS))

for i in range(MONTE_CARLO_RUNS):
    z = np.random.normal(size=HORIZON_DAYS)
    garch_noise = z * cond_std
    mc_matrix[i] = np.exp(combined_log + garch_noise)

combined_low_mc = np.percentile(mc_matrix, 2.5, axis=0)
combined_high_mc = np.percentile(mc_matrix, 97.5, axis=0)

print("✅ Monte Carlo complete")

result_df = pd.DataFrame({
    "combined_price": combined_price,
    "combined_low_mc": combined_low_mc,
    "combined_high_mc": combined_high_mc
}, index=future_dates)

# =========================
# 6) Directional accuracy
# =========================
print("📊 Computing test performance...")

pred_log_test = fc.reindex(X_test.index)["yhat"].values + pred_res_test
pred_price_test = np.exp(pred_log_test)
true_price_test = df.loc[X_test.index, "close"].values

pred_dir = np.diff(pred_price_test) > 0
true_dir = np.diff(true_price_test) > 0

dir_acc = accuracy_score(true_dir, pred_dir)
p_value = binomtest(k=(pred_dir == true_dir).sum(), n=len(pred_dir), p=0.5, alternative='greater').pvalue

print(f"✅ Directional accuracy: {dir_acc:.4f} ({dir_acc * 100:.2f}%)")
print(f"✅ p-value: {p_value:.6f}")

# =========================
# 7) Plot forecast
# =========================
print("📊 Generating forecast plot...")

plt.figure(figsize=(14, 7))
plt.plot(df.index[-180:], df["close"].iloc[-180:], label="Historical", linewidth=2)
plt.plot(result_df.index, result_df["combined_price"], label="Forecast", linewidth=2, color="red")
plt.fill_between(result_df.index, result_df["combined_low_mc"], result_df["combined_high_mc"],
                 alpha=0.2, color="red", label="95% CI")
plt.title(f"BTC Forecast (Optimal Cutoff: {OPTIMAL_START})", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hybrid_forecast_montecarlo.png", dpi=150, bbox_inches="tight")
print("✅ Forecast saved")
plt.show()

# =========================
# 7.5) Learning Curve
# =========================
print("⚙️ Computing learning curve...")

train_sizes = np.linspace(0.2, 1.0, 5)
train_acc, test_acc = [], []

prophet_train_log = fc.reindex(X_train.index)["yhat"].values
prophet_test_log = fc.reindex(X_test.index)["yhat"].values

for frac in train_sizes:
    n = int(frac * len(X_train))
    dtrain_part = xgb.DMatrix(X_train.iloc[:n], label=y_train.iloc[:n])
    bst_part = xgb.train(xgb_params, dtrain_part, num_boost_round=200, verbose_eval=False)

    pred_train_price = np.exp(prophet_train_log[:n] + bst_part.predict(dtrain_part))
    true_train_price = df.loc[X_train.index[:n], "close"].values

    if len(true_train_price) > 1:
        train_acc.append(accuracy_score(np.diff(true_train_price) > 0, np.diff(pred_train_price) > 0))
    else:
        train_acc.append(0.5)

    pred_test_price = np.exp(prophet_test_log + bst_part.predict(xgb.DMatrix(X_test)))
    test_acc.append(accuracy_score(np.diff(true_price_test) > 0, np.diff(pred_test_price) > 0))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes * 100, train_acc, marker="o", label="Train", linewidth=2)
plt.plot(train_sizes * 100, test_acc, marker="o", label="Test", linewidth=2)
plt.axhline(0.5, linestyle="--", color="gray", label="Random")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Directional Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
print("✅ Learning curve saved")
plt.show()

# =========================
# 8) Output
# =========================
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("\n" + "=" * 80)
print("🔮 FORECAST RESULTS")
print("=" * 80)
print(f"\n🎯 Optimal Cutoff: {OPTIMAL_START}")
print(f"   Training days: {len(df)}")
print(f"\n📊 Test Performance ({TEST_LAST_DAYS} days):")
print(f"   • Directional Accuracy: {dir_acc:.4f} ({dir_acc * 100:.2f}%)")
print(f"   • p-value: {p_value:.6f}")

print(f"\n🔮 Forecast Summary:")
print(f"   • Current Price: ${df['close'].iloc[-1]:,.2f}")
print(f"   • Forecast (30d): ${result_df['combined_price'].iloc[29]:,.2f}")
print(f"   • Forecast (90d): ${result_df['combined_price'].iloc[89]:,.2f}")
print(f"   • Forecast (365d): ${result_df['combined_price'].iloc[364]:,.2f}")

print("\n📄 First 10 days:")
print(result_df.head(10))

result_df.to_csv("nextgen_hybrid_forecast_results_montecarlo.csv")
print("\n✅ Saved to: nextgen_hybrid_forecast_results_montecarlo.csv")
print("✅ Plots: pacf_diagnostic.png, hybrid_forecast_montecarlo.png, learning_curve.png")
print("\n" + "=" * 80)
print("✅ ALL DONE!")
print("=" * 80)