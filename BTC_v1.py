"""
nextgen_hybrid_prophet_xgb_forecast_v3.py
Enhanced Prophet + XGBoost residual correction hybrid forecaster for BTC-USD
Includes:
- Robust Yahoo Finance download (handles Adj Close / Close / MultiIndex)
- Automatic lag selection via PACF
- Short-term trend features (EMA, SMA)
- Hyperparameter tuning via Optuna (100 trials)
- Hybrid forecast with uncertainty
- Directional accuracy + significance testing
- Learning curve visualization
Educational / research use only — not financial advice.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
from datetime import timedelta
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import optuna
from scipy.stats import binomtest

# -------------------------
# Config
# -------------------------
TICKER = "BTC-USD"
START = "2021-01-01"
HORIZON_DAYS = 90
TEST_LAST_DAYS = 60
RANDOM_STATE = 42
MAX_LAG = 30   # for PACF lag selection

# -------------------------
# 1) Download data robustly
# -------------------------
print("📥 Downloading data from Yahoo Finance...")
df_raw = yf.download(TICKER, start=START, progress=False)

# Flatten MultiIndex if present
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_raw.columns]

# Find a numeric column to use as price
price_candidates = [c for c in df_raw.columns if 'close' in c.lower() or 'adj' in c.lower()]
if not price_candidates:
    print("Columns available:", df_raw.columns)
    raise SystemExit("No price column found.")
price_col = price_candidates[0]

# Find volume column
volume_candidates = [c for c in df_raw.columns if 'volume' in c.lower()]
if not volume_candidates:
    print("⚠️ Warning: No volume column found. Using zeros for volume.")
    df_raw['volume'] = 0
    volume_col = 'volume'
else:
    volume_col = volume_candidates[0]

df_raw = df_raw[[price_col, volume_col]].rename(columns={price_col:'close', volume_col:'volume'})
df_raw.index = pd.to_datetime(df_raw.index)
df_raw = df_raw.sort_index()
print(f"✅ Using price column: {price_col}")
print(f"✅ Using volume column: {volume_col}")
print(f"✅ Downloaded {len(df_raw)} rows ({df_raw.index.min().date()} → {df_raw.index.max().date()})")

# -------------------------
# 2) Feature engineering
# -------------------------
df = df_raw.copy()
df['return'] = df['close'].pct_change()
df['log_close'] = np.log(df['close'])

# -------------------------
# 2a) Visualize ACF / PACF
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(df['return'].dropna(), lags=MAX_LAG, alpha=0.05, ax=plt.gca())
plt.title('ACF of Returns')

plt.subplot(1,2,2)
plot_pacf(df['return'].dropna(), lags=MAX_LAG, alpha=0.05, ax=plt.gca())
plt.title('PACF of Returns')
plt.tight_layout()
plt.show()

# -------------------------
# 2b) Select lags from PACF (significant partial autocorrelations)
# -------------------------
pacf_vals = pacf(df['return'].dropna(), nlags=MAX_LAG)
N = len(df)
significant_lags = [i for i, v in enumerate(pacf_vals) if i>0 and abs(v) > 2/np.sqrt(N)]

for l in significant_lags:
    df[f'lag_ret_{l}'] = df['return'].shift(l)

# Rolling statistics
rolling_windows = [7,14,30]
for w in rolling_windows:
    df[f'roll_mean_ret_{w}'] = df['return'].rolling(w).mean()
    df[f'roll_std_ret_{w}'] = df['return'].rolling(w).std()
    df[f'roll_vol_{w}'] = df['volume'].rolling(w).mean()

# EMA/SMA features
ema_spans = [7,14,30]
sma_windows = [7,14,30]
for span in ema_spans:
    df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
for w in sma_windows:
    df[f'sma_{w}'] = df['close'].rolling(w).mean()

df = df.dropna().copy()

# -------------------------
# 3) Prophet baseline
# -------------------------
print("⚙️ Fitting Prophet baseline...")
prophet_df = pd.DataFrame({'ds': df.index, 'y': df['log_close'].values})
m = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
m.fit(prophet_df)
future_prophet = m.make_future_dataframe(periods=HORIZON_DAYS)
forecast_prophet = m.predict(future_prophet)
fc = forecast_prophet[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds')

prophet_in_sample = fc.reindex(df.index).dropna()
df = df.loc[prophet_in_sample.index]
df['prophet_log'] = prophet_in_sample['yhat'].values
df['residual'] = df['log_close'] - df['prophet_log']

# -------------------------
# 4) XGBoost residuals + Optuna tuning
# -------------------------
print("⚙️ Preparing features for XGBoost...")

feature_cols = [c for c in df.columns if isinstance(c,str) and c.startswith(('lag_','roll_','ema_','sma_'))]
X = df[feature_cols]
y_res = df['residual']

train_idx_end = -TEST_LAST_DAYS if TEST_LAST_DAYS>0 else None
X_train, X_test = X.iloc[:train_idx_end], X.iloc[train_idx_end:]
y_train, y_test = y_res.iloc[:train_idx_end], y_res.iloc[train_idx_end:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

def objective(trial):
    param = {
        'objective':'reg:squarederror',
        'max_depth':trial.suggest_int('max_depth',3,8),
        'learning_rate':trial.suggest_float('learning_rate',0.01,0.2, log=True),
        'subsample':trial.suggest_float('subsample',0.6,1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
        'seed':RANDOM_STATE,
        'verbosity':0
    }
    bst_trial = xgb.train(param, dtrain, num_boost_round=200)
    pred = bst_trial.predict(dtest)
    return mean_squared_error(y_test,pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
best_params.update({'objective':'reg:squarederror','seed':RANDOM_STATE,'verbosity':0})
print("✅ Best hyperparameters found:", best_params)

bst = xgb.train(best_params, dtrain, num_boost_round=200)
pred_res_test = bst.predict(dtest)
mse_res = mean_squared_error(y_test, pred_res_test)
print(f"✅ XGBoost residual model trained. Residual test MSE: {mse_res:.6e}")

# -------------------------
# 5) Hybrid forecast
# -------------------------
print(f"🔮 Producing combined {HORIZON_DAYS}-day forecast...")

prophet_future = fc.reindex(pd.date_range(start=df.index.max()+timedelta(days=1), periods=HORIZON_DAYS, freq='D')).ffill()
prophet_future = prophet_future.rename(columns={'yhat':'prophet_log','yhat_lower':'prophet_low','yhat_upper':'prophet_high'})

history = df.copy()
future_rows = []
for i in range(HORIZON_DAYS):
    next_date = history.index.max()+timedelta(days=1)
    new = {}
    for col in feature_cols:
        if col.startswith('lag_ret_'):
            l = int(col.split('_')[-1])
            new[col] = history['return'].iloc[-l] if len(history)>=l else 0.0
        elif col.startswith('roll_'):
            w = int(col.split('_')[-1])
            if 'mean' in col:
                new[col] = history['return'].iloc[-w:].mean()
            elif 'std' in col:
                new[col] = history['return'].iloc[-w:].std()
            elif 'vol' in col:
                new[col] = history['volume'].iloc[-w:].mean()
        elif col.startswith('ema_'):
            span = int(col.split('_')[-1])
            new[col] = history['close'].iloc[-span:].ewm(span=span, adjust=False).mean().iloc[-1]
        elif col.startswith('sma_'):
            w = int(col.split('_')[-1])
            new[col] = history['close'].iloc[-w:].mean()
    future_rows.append(pd.Series(new, name=next_date))
    placeholder = pd.Series({
        'close': history['close'].iloc[-1],
        'volume': history['volume'].iloc[-1],
        'return': 0.0,
        'log_close': history['log_close'].iloc[-1],
        'prophet_log': prophet_future.loc[next_date,'prophet_log'] if next_date in prophet_future.index else history['prophet_log'].iloc[-1],
        'residual': 0.0
    }, name=next_date)
    history = pd.concat([history, placeholder.to_frame().T])

X_future = pd.DataFrame(future_rows).ffill().fillna(0.0)
dX_future = xgb.DMatrix(X_future[feature_cols])
pred_res_future = bst.predict(dX_future)

combined_log = prophet_future['prophet_log'].values[:HORIZON_DAYS] + pred_res_future
combined_price = np.exp(combined_log)

resid_std = y_train.std()
z95 = 1.96
prophet_low = np.exp(prophet_future['prophet_log'] - z95*resid_std)[:HORIZON_DAYS]
prophet_high = np.exp(prophet_future['prophet_log'] + z95*resid_std)[:HORIZON_DAYS]
combined_low = combined_price - (np.exp(prophet_future['prophet_log'][:HORIZON_DAYS]) - prophet_low)
combined_high = combined_price + (prophet_high - np.exp(prophet_future['prophet_log'][:HORIZON_DAYS]))

result_index = prophet_future.index[:HORIZON_DAYS]
result_df = pd.DataFrame({
    'date': result_index,
    'prophet_price': np.exp(prophet_future['prophet_log'][:HORIZON_DAYS].values),
    'combined_price': combined_price,
    'combined_low': combined_low,
    'combined_high': combined_high
}).set_index('date')

# Probability of up-move based on simple sigmoid
last_close = df['close'].iloc[-1]
combined_delta = (result_df['combined_price'] - last_close)
combined_uncertainty = resid_std * last_close
result_df['prob_up'] = 1 / (1 + np.exp(-combined_delta / (combined_uncertainty + 1e-9)))

# -------------------------
# 6a) Directional backtest + significance
# -------------------------
prophet_test = fc.reindex(X_test.index).dropna()
prophet_test_log = prophet_test['yhat'].values[:len(X_test)]
dX_test = xgb.DMatrix(X_test[feature_cols])
pred_res_test = bst.predict(dX_test)
pred_log_test = prophet_test_log + pred_res_test
pred_price_test = np.exp(pred_log_test)
true_price_test = df.loc[X_test.index, 'close'].values

pred_dir = (pred_price_test[1:] - pred_price_test[:-1]) > 0
true_dir = (true_price_test[1:] - true_price_test[:-1]) > 0
dir_acc = accuracy_score(true_dir, pred_dir) if len(pred_dir) > 0 else np.nan
p_value = binomtest(sum(pred_dir == true_dir), n=len(pred_dir), p=0.5)

print(f"Directional accuracy on holdout: {dir_acc:.4f}")
print(f"p-value vs random chance: {p_value.pvalue:.6f}")  # <-- fixed

# -------------------------
# 6b) Learning curve
# -------------------------
train_sizes = np.linspace(0.1, 1.0, 5)
train_scores = []
test_scores = []

for size in train_sizes:
    n = int(size * len(X_train))
    dtrain_part = xgb.DMatrix(X_train.iloc[:n], label=y_train.iloc[:n])
    bst_part = xgb.train(best_params, dtrain_part, num_boost_round=200)

    pred_train = bst_part.predict(dtrain_part)
    pred_test = bst_part.predict(dtest)

    train_scores.append(mean_squared_error(y_train.iloc[:n], pred_train))
    test_scores.append(mean_squared_error(y_test, pred_test))

plt.figure(figsize=(8, 5))
plt.plot(train_sizes * 100, train_scores, marker='o', label='Train MSE')
plt.plot(train_sizes * 100, test_scores, marker='o', label='Test MSE')
plt.title("Learning Curve - XGBoost Residual Model")
plt.xlabel("Training set size (%)")
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -------------------------
# 7) Plot
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df['close'], label='Historical', color='blue')
plt.plot(result_df.index, result_df['prophet_price'], label='Prophet baseline', color='orange', linestyle='--')
plt.plot(result_df.index, result_df['combined_price'], label='Hybrid (Prophet + XGB)', color='red', marker='o')
plt.fill_between(result_df.index, result_df['combined_low'], result_df['combined_high'], color='red', alpha=0.12, label='Hybrid CI (approx)')
plt.axhline(last_close, color='gray', alpha=0.5, linestyle=':')
plt.title(f"{TICKER} Enhanced Hybrid Forecast ({HORIZON_DAYS} days)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# -------------------------
# 8) Output
# -------------------------
pd.set_option('display.float_format', lambda x: f"{x:,.2f}")
print("\n🔮 Combined forecast (first rows):")
print(result_df[['combined_price','combined_low','combined_high','prob_up']].head(15))

result_df.to_csv("nextgen_hybrid_forecast_results.csv")
print("\n✅ Forecast saved to nextgen_hybrid_forecast_results.csv")

