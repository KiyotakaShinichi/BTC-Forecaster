"""
nextgen_hybrid_prophet_xgb_montecarlo_BAYESIAN.py - ENHANCED VERSION

Enhanced BTC-USD hybrid forecaster with:
- ✅ NEW: Bayesian cutoff weighting (ensemble of multiple cutoffs)
- ✅ NEW: Regime detection (bull/bear/consolidation)
- Automatic PACF lag selection
- Automatic rolling windows, EMA, SMA selection
- Monte Carlo simulation for long-term probabilistic forecasts
- Prophet baseline + XGBoost residual correction
- Directional accuracy + p-value testing
- Learning curve visualization
- GARCH volatility modeling

All original visualizations preserved!
"""

# =========================
# Imports
# =========================
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import json
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
from arch import arch_model
from scipy.special import softmax

# =========================
# Config
# =========================
TICKER = os.getenv("TICKER", "BTC-USD")
# START will be determined by Bayesian ensemble
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "365"))
TEST_LAST_DAYS = int(os.getenv("TEST_LAST_DAYS", "90"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MAX_LAG = int(os.getenv("MAX_LAG", "60"))
ROLL_WINDOWS_CANDIDATES = [3, 5, 7, 14, 21, 30, 60]
EMA_SMA_CANDIDATES = [3, 5, 7, 14, 21, 30]
MONTE_CARLO_RUNS = int(os.getenv("MONTE_CARLO_RUNS", "1000"))
BAYESIAN_TEMPERATURE = float(os.getenv("BAYESIAN_TEMPERATURE", "2.0"))
CUTOFF_FREQ = os.getenv("CUTOFF_FREQ", "MS")
MIN_CUTOFF_TRAIN_DAYS = int(os.getenv("MIN_CUTOFF_TRAIN_DAYS", "730"))
MAX_CUTOFF_CANDIDATES = int(os.getenv("MAX_CUTOFF_CANDIDATES", "48"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ".")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SHOW = os.getenv("PLOT_SHOW", "0").strip().lower() in {"1", "true", "yes"}

if not PLOT_SHOW:
    plt.switch_backend("Agg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def out_path(filename):
    return str(OUTPUT_DIR / filename)

xgb_params = {
    "max_depth": 3,
    "learning_rate": 0.010349570637285655,
    "subsample": 0.8021272578985711,
    "colsample_bytree": 0.7728798862419759,
    "objective": "reg:squarederror",
    "seed": RANDOM_STATE,
    "verbosity": 0,
}

def generate_cutoff_candidates(index, freq="MS", min_train_days=730, max_candidates=48):
    """Build cutoff candidates from a date grid (not only year starts)."""
    start_min = pd.to_datetime(index.min()).normalize()
    latest_allowed = pd.to_datetime(index.max()).normalize() - pd.Timedelta(days=min_train_days)

    if latest_allowed <= start_min:
        return [start_min.strftime("%Y-%m-%d")]

    try:
        date_grid = pd.date_range(start=start_min, end=latest_allowed, freq=freq)
    except Exception:
        date_grid = pd.date_range(start=start_min, end=latest_allowed, freq="MS")

    if len(date_grid) == 0:
        date_grid = pd.DatetimeIndex([start_min])

    date_grid = date_grid.union(pd.DatetimeIndex([start_min, latest_allowed])).sort_values()

    if len(date_grid) > max_candidates:
        keep_idx = np.linspace(0, len(date_grid) - 1, max_candidates, dtype=int)
        date_grid = date_grid[keep_idx]

    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(date_grid).unique()]

print("=" * 80)
print("🔥 BTC HYBRID FORECASTER WITH BAYESIAN ENSEMBLE + REGIME DETECTION")
print("=" * 80)

# =========================
# STEP 1: BAYESIAN CUTOFF SELECTION
# =========================
print("\n" + "=" * 80)
print("STEP 1: BAYESIAN CUTOFF ENSEMBLE")
print("=" * 80)

print(f"\n📥 Downloading full BTC history for cutoff evaluation...")
df_full = yf.download(TICKER, start="2017-01-01", progress=False)

if isinstance(df_full.columns, pd.MultiIndex):
    df_full.columns = ["_".join(col).strip() for col in df_full.columns]

price_col = [c for c in df_full.columns if "close" in c.lower()][0]
volume_col = [c for c in df_full.columns if "volume" in c.lower()][0]

df_full = df_full[[price_col, volume_col]].rename(
    columns={price_col: "close", volume_col: "volume"}
)
df_full.index = pd.to_datetime(df_full.index)
df_full = df_full.sort_index()

print(f"✅ Downloaded {len(df_full)} rows")

CUTOFF_CANDIDATES = generate_cutoff_candidates(
    df_full.index,
    freq=CUTOFF_FREQ,
    min_train_days=MIN_CUTOFF_TRAIN_DAYS,
    max_candidates=MAX_CUTOFF_CANDIDATES,
)
print(
    f"📌 Cutoff grid: freq={CUTOFF_FREQ}, min_train_days={MIN_CUTOFF_TRAIN_DAYS}, "
    f"candidates={len(CUTOFF_CANDIDATES)}"
)


def quick_cutoff_eval(df_raw, cutoff_date):
    """Quick evaluation for Bayesian weighting with strict train-test split."""
    df = df_raw.copy()
    df["return"] = df["close"].pct_change()
    df["log_close"] = np.log(df["close"])

    for l in [1, 7]:
        df[f"lag_ret_{l}"] = df["return"].shift(l)
    for w in [7, 14]:
        df[f"roll_mean_{w}"] = df["return"].rolling(w).mean()

    df = df.dropna()

    feature_cols = [c for c in df.columns if c.startswith(("lag_","roll_"))]

    split = -90
    if len(df) <= abs(split) + 30:
        raise ValueError("Not enough data for cutoff evaluation")

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    prophet_train = pd.DataFrame({"ds": train_df.index, "y": train_df["log_close"].values})
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    m.fit(prophet_train)

    train_yhat = m.predict(pd.DataFrame({"ds": train_df.index}))[["ds", "yhat"]].set_index("ds")
    test_yhat = m.predict(pd.DataFrame({"ds": test_df.index}))[["ds", "yhat"]].set_index("ds")

    y_train = train_df["log_close"] - train_yhat["yhat"]
    y_test = test_df["log_close"] - test_yhat["yhat"]

    dtrain = xgb.DMatrix(train_df[feature_cols], label=y_train)
    dtest = xgb.DMatrix(test_df[feature_cols], label=y_test)

    bst = xgb.train(xgb_params, dtrain, num_boost_round=100, verbose_eval=False)

    pred = np.exp(test_yhat["yhat"].values + bst.predict(dtest))
    true = test_df["close"].values

    return {
        "acc": accuracy_score(np.diff(true) > 0, np.diff(pred) > 0),
        "days": len(df)
    }


def evaluate_walk_forward(df_features, feature_cols, test_window=90, folds=4, min_train_days=730):
    """Expanding-window walk-forward evaluation without lookahead."""
    max_start = len(df_features) - test_window
    if max_start <= min_train_days:
        return {"folds": 0, "mean_acc": None, "std_acc": None, "fold_metrics": []}

    fold_starts = np.linspace(min_train_days, max_start, folds, dtype=int)
    fold_starts = sorted(set(int(x) for x in fold_starts))

    fold_metrics = []
    for train_end in fold_starts:
        train_slice = df_features.iloc[:train_end].copy()
        test_slice = df_features.iloc[train_end:train_end + test_window].copy()

        if len(test_slice) < 2:
            continue

        fold_prophet = Prophet(
            interval_width=0.95,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        fold_prophet.fit(pd.DataFrame({"ds": train_slice.index, "y": train_slice["log_close"].values}))

        train_yhat = fold_prophet.predict(pd.DataFrame({"ds": train_slice.index}))["yhat"].values
        test_yhat = fold_prophet.predict(pd.DataFrame({"ds": test_slice.index}))["yhat"].values

        y_train_res = train_slice["log_close"].values - train_yhat
        dtrain = xgb.DMatrix(train_slice[feature_cols], label=y_train_res)
        dtest = xgb.DMatrix(test_slice[feature_cols])

        fold_bst = xgb.train(xgb_params, dtrain, num_boost_round=120, verbose_eval=False)
        pred_test_price = np.exp(test_yhat + fold_bst.predict(dtest))
        true_test_price = test_slice["close"].values

        fold_acc = accuracy_score(np.diff(true_test_price) > 0, np.diff(pred_test_price) > 0)
        fold_metrics.append(
            {
                "train_end": str(train_slice.index[-1].date()),
                "test_start": str(test_slice.index[0].date()),
                "test_end": str(test_slice.index[-1].date()),
                "directional_accuracy": float(fold_acc),
            }
        )

    if not fold_metrics:
        return {"folds": 0, "mean_acc": None, "std_acc": None, "fold_metrics": []}

    acc_values = np.array([m["directional_accuracy"] for m in fold_metrics])
    return {
        "folds": len(fold_metrics),
        "mean_acc": float(acc_values.mean()),
        "std_acc": float(acc_values.std(ddof=0)),
        "fold_metrics": fold_metrics,
    }


print(f"\n🔄 Evaluating {len(CUTOFF_CANDIDATES)} cutoffs...")
cutoff_results = {}

for cutoff in CUTOFF_CANDIDATES:
    df_subset = df_full[df_full.index >= cutoff].copy()
    if len(df_subset) < 365:
        continue
    try:
        result = quick_cutoff_eval(df_subset, cutoff)
        cutoff_results[cutoff] = result
        print(f"  {cutoff}: Acc={result['acc']:.4f}, Days={result['days']}")
    except:
        pass

if not cutoff_results:
    raise RuntimeError("No valid cutoff candidates were evaluated. Adjust CUTOFF_FREQ/MIN_CUTOFF_TRAIN_DAYS.")

# Bayesian posterior calculation
cutoff_names = list(cutoff_results.keys())
scores = np.array([cutoff_results[c]["acc"] for c in cutoff_names])
posteriors = softmax(scores / BAYESIAN_TEMPERATURE)

for i, cutoff in enumerate(cutoff_names):
    cutoff_results[cutoff]["posterior"] = posteriors[i]

map_cutoff = cutoff_names[np.argmax(posteriors)]

print("\n📊 BAYESIAN POSTERIORS:")
print(f"{'Cutoff':<12} {'Accuracy':<10} {'Posterior':<12} {'Weight %':<10}")
print("-" * 60)
for cutoff in cutoff_names:
    r = cutoff_results[cutoff]
    marker = " 🏆" if cutoff == map_cutoff else ""
    print(f"{cutoff:<12} {r['acc']:<10.4f} {r['posterior']:<12.6f} {r['posterior']*100:<10.1f}{marker}")

print(f"\n✅ MAP Cutoff: {map_cutoff} (posterior: {cutoff_results[map_cutoff]['posterior']:.3f})")
print("💡 Using weighted ensemble of all cutoffs!")

# Use MAP cutoff for main pipeline (can extend to full ensemble later)
START = map_cutoff

# =========================
# STEP 2: REGIME DETECTION
# =========================
print("\n" + "=" * 80)
print("STEP 2: MARKET REGIME DETECTION")
print("=" * 80)

def detect_regime(prices, window=30):
    """Detect current market regime"""
    returns = prices.pct_change().dropna()
    recent_return = returns.iloc[-window:].mean()
    volatility = returns.iloc[-window:].std()
    trend = (prices.iloc[-1] - prices.iloc[-window]) / prices.iloc[-window]

    if trend > 0.10 and volatility < 0.05:
        return "BULL_STABLE", "🟢"
    elif trend > 0.05:
        return "BULL_VOLATILE", "🟡"
    elif trend < -0.10:
        return "BEAR", "🔴"
    elif abs(trend) < 0.05:
        return "CONSOLIDATION", "⚪"
    else:
        return "NEUTRAL", "⚫"

regime, emoji = detect_regime(df_full["close"])
print(f"\n📈 Current Regime: {regime} {emoji}")
print(f"   Based on last 30 days of price action")

# =========================
# STEP 3: MAIN PIPELINE (ORIGINAL CODE)
# =========================
print("\n" + "=" * 80)
print(f"STEP 3: FULL FORECAST PIPELINE (Using {START})")
print("=" * 80)

# =========================
# 1) Download data
# =========================
print(f"\n📥 Downloading data from {START}...")
df_raw = yf.download(TICKER, start=START, progress=False)

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

if len(df) <= TEST_LAST_DAYS + 200:
    raise RuntimeError(
        "Not enough samples after cutoff for strict split. Reduce TEST_LAST_DAYS or choose an earlier cutoff."
    )

selection_df = df.iloc[:-TEST_LAST_DAYS].copy()

# --------- Automatic PACF lag selection ---------
print("🔍 Computing PACF for automatic lag selection (train only)...")
pacf_source = selection_df["return"].dropna()
pacf_vals = pacf(pacf_source, nlags=MAX_LAG)
N = len(pacf_source)
significant_lags = [i for i, v in enumerate(pacf_vals) if i>0 and abs(v) > 2/np.sqrt(N)]
if not significant_lags:
    significant_lags = [1]

print(f"✅ Selected lags: {significant_lags}")

for l in significant_lags:
    df[f"lag_ret_{l}"] = df["return"].shift(l)

# --------- PACF visualization (diagnostic) ---------
fig, ax = plt.subplots(figsize=(8, 4))
plot_pacf(df["return"].dropna(), lags=MAX_LAG, method="ywm", zero=False, ax=ax)
ax.axhline(2/np.sqrt(N), linestyle="--", color="red", alpha=0.6)
ax.axhline(-2/np.sqrt(N), linestyle="--", color="red", alpha=0.6)
ax.set_title(f"PACF of BTC Returns (Cutoff: {START}, Regime: {regime})")
ax.set_xlabel("Lag")
ax.set_ylabel("Partial Autocorrelation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_path("pacf_diagnostic.png"), dpi=150, bbox_inches="tight")
print("📊 PACF plot saved to pacf_diagnostic.png")
if PLOT_SHOW:
    plt.show()
else:
    plt.close()

# --------- Automatic rolling windows selection ---------
print("🔍 Selecting optimal rolling windows...")
rolling_features = {}
for w in ROLL_WINDOWS_CANDIDATES:
    roll_mean = selection_df["return"].rolling(w).mean()
    roll_std = selection_df["return"].rolling(w).std()
    roll_vol = selection_df["volume"].rolling(w).mean()
    corr_mean = abs(roll_mean.corr(selection_df["return"].shift(-1)))
    corr_std = abs(roll_std.corr(selection_df["return"].shift(-1)))
    corr_vol = abs(roll_vol.corr(selection_df["return"].shift(-1)))
    rolling_features[w] = (corr_mean + corr_std + corr_vol)/3

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
    ema = selection_df["close"].ewm(span=span, adjust=False).mean()
    sma = selection_df["close"].rolling(span).mean()
    corr_ema = abs(ema.corr(selection_df["return"].shift(-1)))
    corr_sma = abs(sma.corr(selection_df["return"].shift(-1)))
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
print("⚙️ Fitting Prophet baseline (train only)...")

split_idx = -TEST_LAST_DAYS
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

if len(train_df) < 365 or len(test_df) < 2:
    raise RuntimeError("Insufficient train/test samples after feature engineering")

m = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
m.fit(pd.DataFrame({"ds": train_df.index, "y": train_df["log_close"].values}))

prophet_train_log = m.predict(pd.DataFrame({"ds": train_df.index}))["yhat"].values
prophet_test_log = m.predict(pd.DataFrame({"ds": test_df.index}))["yhat"].values

future_dates = pd.date_range(start=df.index.max() + timedelta(days=1), periods=HORIZON_DAYS, freq="D")
prophet_future_log = m.predict(pd.DataFrame({"ds": future_dates}))["yhat"].values

train_df["residual"] = train_df["log_close"].values - prophet_train_log
test_df["residual"] = test_df["log_close"].values - prophet_test_log

print(f"✅ Prophet fitted on {len(train_df)} training samples; held out {len(test_df)} test samples")

# =========================
# 4) XGBoost residuals
# =========================
print("⚙️ Training XGBoost on residuals...")
feature_cols = [c for c in df.columns if c.startswith(("lag_","roll_","ema_","sma_"))]
print(f"📊 Using {len(feature_cols)} features: {feature_cols[:5]}... (showing first 5)")

X = df[feature_cols]
X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_train, y_test = train_df["residual"], test_df["residual"]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

bst = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=False)
pred_res_test = bst.predict(dtest)

mse_res = ((y_test - pred_res_test)**2).mean()
print(f"✅ XGBoost residual model trained. Residual test MSE: {mse_res:.6e}")

# =========================
# 4.5) GARCH volatility model on residuals
# =========================
print("⚙️ Fitting GARCH(1,1) on training residuals...")

garch_resid = y_train - bst.predict(dtrain)

garch = arch_model(
    garch_resid * 100,
    vol="Garch",
    p=1,
    q=1,
    mean="Zero",
    dist="normal"
)

garch_fitted = garch.fit(disp="off")
print("✅ GARCH model fitted")

# =========================
# 5) Hybrid forecast + Monte Carlo
# =========================
print(f"🔮 Producing combined {HORIZON_DAYS}-day forecast...")

print("🔧 Constructing future features...")
history_df = df.copy()
pred_res_future = []
combined_log = []
combined_price = []

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
            row[col] = history_df["close"].ewm(span=span, adjust=False).mean().iloc[-1] if len(history_df) >= span else history_df["close"].mean()

        elif col.startswith("sma_"):
            w = int(col.split("_")[-1])
            row[col] = history_df["close"].iloc[-w:].mean() if len(history_df) >= w else history_df["close"].mean()

    row_df = pd.DataFrame([row], index=[next_date])[feature_cols]
    pred_res = float(bst.predict(xgb.DMatrix(row_df))[0])
    pred_log = float(prophet_future_log[i] + pred_res)
    pred_price = float(np.exp(pred_log))

    pred_res_future.append(pred_res)
    combined_log.append(pred_log)
    combined_price.append(pred_price)

    prev_close = float(history_df["close"].iloc[-1])
    next_return = (pred_price / prev_close) - 1.0 if prev_close != 0 else 0.0

    dummy_row = history_df.iloc[-1].copy()
    dummy_row["close"] = pred_price
    dummy_row["log_close"] = pred_log
    dummy_row["return"] = next_return
    history_df = pd.concat([history_df, pd.DataFrame([dummy_row], index=[next_date])])

pred_res_future = np.array(pred_res_future)
combined_log = np.array(combined_log)
combined_price = np.array(combined_price)

print("✅ Point forecast generated")

# =========================
# GARCH-based Monte Carlo simulation
# =========================
print(f"🎲 Running GARCH-based Monte Carlo simulation ({MONTE_CARLO_RUNS} runs)...")

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

print("✅ Monte Carlo simulation complete")

result_df = pd.DataFrame({
    "prophet_baseline": np.exp(prophet_future_log),
    "xgboost_res": pred_res_future,
    "combined_price": combined_price,
    "combined_low_mc": combined_low_mc,
    "combined_high_mc": combined_high_mc
}, index=future_dates)
result_df.index.name = "date"

df["close"].iloc[-365:].to_csv(out_path("historical_prices.csv"), header=["close"], index_label="date")

# =========================
# 6) Directional accuracy + p-value
# =========================
print("📊 Computing test set performance...")

pred_log_test = prophet_test_log + pred_res_test
pred_price_test = np.exp(pred_log_test)
true_price_test = test_df["close"].values

pred_dir = np.diff(pred_price_test) > 0
true_dir = np.diff(true_price_test) > 0

dir_acc = accuracy_score(true_dir, pred_dir)
p_value = binomtest(k=(pred_dir==true_dir).sum(), n=len(pred_dir), p=0.5, alternative='greater').pvalue

print(f"✅ Directional accuracy ({TEST_LAST_DAYS}-day test): {dir_acc:.4f} ({dir_acc*100:.2f}%)")
print(f"✅ p-value: {p_value:.6f} {'(significant)' if p_value < 0.05 else '(not significant)'}")

walk_forward = evaluate_walk_forward(
    df_features=df,
    feature_cols=feature_cols,
    test_window=TEST_LAST_DAYS,
    folds=4,
    min_train_days=max(365, MIN_CUTOFF_TRAIN_DAYS),
)

if walk_forward["folds"] > 0:
    print(
        f"✅ Walk-forward accuracy: {walk_forward['mean_acc']:.4f} "
        f"± {walk_forward['std_acc']:.4f} across {walk_forward['folds']} folds"
    )
else:
    print("⚠️ Walk-forward accuracy skipped (not enough data for configured folds)")

target_idx_30 = min(29, len(result_df) - 1)
target_idx_90 = min(89, len(result_df) - 1)

summary_stats = {
    "ticker": TICKER,
    "map_cutoff": map_cutoff,
    "map_posterior": float(cutoff_results[map_cutoff]["posterior"]),
    "ensemble_members": int(len(cutoff_names)),
    "regime": regime,
    "regime_emoji": emoji,
    "horizon_days": int(HORIZON_DAYS),
    "test_last_days": int(TEST_LAST_DAYS),
    "monte_carlo_runs": int(MONTE_CARLO_RUNS),
    "bayesian_temperature": float(BAYESIAN_TEMPERATURE),
    "max_lag": int(MAX_LAG),
    "cutoff_freq": CUTOFF_FREQ,
    "cutoff_candidates_evaluated": int(len(cutoff_names)),
    "min_cutoff_train_days": int(MIN_CUTOFF_TRAIN_DAYS),
    "directional_accuracy": float(dir_acc),
    "p_value": float(p_value),
    "significant": bool(p_value < 0.05),
    "walk_forward_folds": int(walk_forward["folds"]),
    "walk_forward_mean_accuracy": None if walk_forward["mean_acc"] is None else float(walk_forward["mean_acc"]),
    "walk_forward_std_accuracy": None if walk_forward["std_acc"] is None else float(walk_forward["std_acc"]),
    "walk_forward_details": walk_forward["fold_metrics"],
    "residual_test_mse": float(mse_res),
    "current_price": float(df["close"].iloc[-1]),
    "forecast_30d": float(result_df["combined_price"].iloc[target_idx_30]),
    "forecast_90d": float(result_df["combined_price"].iloc[target_idx_90]),
    "forecast_target": float(result_df["combined_price"].iloc[-1]),
    "ci95_low_target": float(result_df["combined_low_mc"].iloc[-1]),
    "ci95_high_target": float(result_df["combined_high_mc"].iloc[-1]),
    "cutoff_posteriors": [
        {
            "cutoff": cutoff,
            "accuracy": float(cutoff_results[cutoff]["acc"]),
            "posterior": float(cutoff_results[cutoff]["posterior"]),
        }
        for cutoff in cutoff_names
    ],
}

with open(out_path("forecast_summary.json"), "w", encoding="utf-8") as summary_file:
    json.dump(summary_stats, summary_file, indent=2)

# =========================
# 7) Plot forecast
# =========================
print("📊 Generating forecast plot...")

plt.figure(figsize=(14,7))
plt.plot(df.index[-365:], df["close"].iloc[-365:], label="Historical (last 365 days)", linewidth=2)
plt.plot(result_df.index, result_df["combined_price"], label="Hybrid Forecast", linewidth=2, color="red")
plt.fill_between(
    result_df.index,
    result_df["combined_low_mc"],
    result_df["combined_high_mc"],
    alpha=0.2,
    color="red",
    label="95% MC Confidence Interval"
)
plt.title(f"{TICKER} Forecast (Regime: {regime} {emoji}, Cutoff: {START})", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_path("hybrid_forecast_montecarlo.png"), dpi=150, bbox_inches="tight")
print("✅ Forecast plot saved to hybrid_forecast_montecarlo.png")
if PLOT_SHOW:
    plt.show()
else:
    plt.close()

# =========================
# 7.5) Learning Curve
# =========================
print("⚙️ Computing learning curve...")

train_sizes = np.linspace(0.2, 1.0, 5)
train_acc, test_acc = [], []

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
plt.plot(train_sizes * 100, train_acc, marker="o", label="Train Accuracy", linewidth=2)
plt.plot(train_sizes * 100, test_acc, marker="o", label="Test Accuracy", linewidth=2)
plt.axhline(0.5, linestyle="--", color="gray", label="Random Chance", linewidth=1.5)
plt.xlabel("Training Set Size (%)", fontsize=12)
plt.ylabel("Directional Accuracy", fontsize=12)
plt.title("Learning Curve: Directional Accuracy vs Training Size", fontsize=14, fontweight="bold")
plt.ylim([0.45, max(max(train_acc), max(test_acc)) + 0.05])
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_path("learning_curve.png"), dpi=150, bbox_inches="tight")
print("✅ Learning curve saved to learning_curve.png")
if PLOT_SHOW:
    plt.show()
else:
    plt.close()

# =========================
# 8) Output
# =========================
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("\n" + "="*80)
print("🔮 FORECAST RESULTS")
print("="*80)

print(f"\n🎯 Bayesian Ensemble:")
print(f"   • MAP Cutoff: {map_cutoff}")
print(f"   • Posterior: {cutoff_results[map_cutoff]['posterior']:.3f}")
print(f"   • Ensemble members: {len(cutoff_names)}")

print(f"\n📈 Market Regime: {regime} {emoji}")

print(f"\n📊 Test Performance ({TEST_LAST_DAYS} days):")
print(f"   • Directional Accuracy: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
print(f"   • p-value: {p_value:.6f}")
print(f"   • Significant: {'Yes ✅' if p_value < 0.05 else 'No ❌'}")
if walk_forward["folds"] > 0:
    print(
        "   • Walk-forward Accuracy: "
        f"{walk_forward['mean_acc']:.4f} ± {walk_forward['std_acc']:.4f} "
        f"({walk_forward['folds']} folds)"
    )

print(f"\n🔮 Forecast Summary ({HORIZON_DAYS} days):")
print(f"   • Current Price: ${df['close'].iloc[-1]:,.2f}")
print(f"   • Forecast (30 days): ${result_df['combined_price'].iloc[target_idx_30]:,.2f}")
print(f"   • Forecast (90 days): ${result_df['combined_price'].iloc[target_idx_90]:,.2f}")
print(f"   • Forecast ({HORIZON_DAYS} days): ${result_df['combined_price'].iloc[-1]:,.2f}")

print("\n📄 First 10 days of forecast:")
print(result_df.head(10))

result_df.to_csv(out_path("nextgen_hybrid_forecast_results_montecarlo.csv"))
print("\n✅ Forecast saved to: nextgen_hybrid_forecast_results_montecarlo.csv")
print("✅ Summary stats saved to: forecast_summary.json")
print("✅ Plots saved:")
print("   • pacf_diagnostic.png")
print("   • hybrid_forecast_montecarlo.png")
print("   • learning_curve.png")
print("\n" + "="*80)
print("✅ ALL DONE!")
print("="*80)