"""
nextgen_hybrid_prophet_xgb_forecast_v4.py
Enhanced Prophet + XGBoost residual correction hybrid forecaster for BTC-USD
Includes:
- Robust Yahoo Finance download
- Automatic lag selection via PACF
- Short-term trend features (EMA, SMA, rolling)
- Hyperparameter tuning via Optuna
- Hybrid forecast with Monte Carlo uncertainty
- Directional accuracy backtest
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
from sklearn.model_selection import TimeSeriesSplit

# -------------------------
# Config
# -------------------------
TICKER = "BTC-USD"
START = "2021-01-01"
HORIZON_DAYS = 90
TEST_LAST_DAYS = 60
RANDOM_STATE = 42
MAX_LAG = 30

# -------------------------
# 1) Download data
# -------------------------
print("📥 Downloading data from Yahoo Finance...")
df_raw = yf.download(TICKER, start=START, progress=False)

# Flatten MultiIndex if present
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_raw.columns]

# Identify price & volume columns
price_candidates = [c for c in df_raw.columns if 'close' in c.lower() or 'adj' in c.lower()]
volume_candidates = [c for c in df_raw.columns if 'volume' in c.lower()]

price_col = price_candidates[0]
volume_col = volume_candidates[0] if volume_candidates else 'volume'
if volume_col == 'volume': df_raw['volume'] = 0

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

# Visualize ACF/PACF
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_acf(df['return'].dropna(), lags=MAX_LAG, alpha=0.05, ax=plt.gca())
plt.title('ACF of Returns')
plt.subplot(1,2,2)
plot_pacf(df['return'].dropna(), lags=MAX_LAG, alpha=0.05, ax=plt.gca())
plt.title('PACF of Returns')
plt.tight_layout()
plt.show()

# PACF-based lag features
pacf_vals = pacf(df['return'].dropna(), nlags=MAX_LAG)
N = len(df)
significant_lags = [i for i, v in enumerate(pacf_vals) if i>0 and abs(v) > 2/np.sqrt(N)]
for l in significant_lags:
    df[f'lag_ret_{l}'] = df['return'].shift(l)

# Rolling features
rolling_windows = [7,14,30]
for w in rolling_windows:
    df[f'roll_mean_ret_{w}'] = df['return'].rolling(w).mean()
    df[f'roll_std_ret_{w}'] = df['return'].rolling(w).std()
    df[f'roll_vol_{w}'] = df['volume'].rolling(w).mean()

# EMA/SMA
ema_spans = [7,14,30]
sma_windows = [7,14,30]
for span in ema_spans: df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
for w in sma_windows: df[f'sma_{w}'] = df['close'].rolling(w).mean()

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
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'seed': RANDOM_STATE,
        'verbosity': 0
    }
    tscv = TimeSeriesSplit(n_splits=4)
    mse_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(param, dtr, num_boost_round=200)
        preds = model.predict(dval)
        mse_scores.append(mean_squared_error(y_val, preds))
    return np.mean(mse_scores)

# Run Optuna
print("⚙️ Tuning XGBoost hyperparameters with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("✅ Best XGB params:", best_params)

bst = xgb.train(
    {**best_params, 'objective':'reg:squarederror', 'seed':RANDOM_STATE},
    dtrain,
    num_boost_round=200
)

# -------------------------
# 5) Hybrid forecast
# -------------------------
print(f"🔮 Producing combined {HORIZON_DAYS}-day forecast...")
history = df.copy()
future_rows = []

for i in range(HORIZON_DAYS):
    next_date = history.index.max() + timedelta(days=1)
    new = {}
    for col in feature_cols:
        if col.startswith('lag_ret_'):
            l = int(col.split('_')[-1])
            new[col] = history['return'].iloc[-l] if len(history)>=l else 0.0
        elif col.startswith('roll_'):
            w = int(col.split('_')[-1])
            if 'mean' in col: new[col] = history['return'].iloc[-w:].mean()
            elif 'std' in col: new[col] = history['return'].iloc[-w:].std()
            elif 'vol' in col: new[col] = history['volume'].iloc[-w:].mean()
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
        'prophet_log': fc['yhat'].reindex([next_date]).values[0] if next_date in fc.index else history['prophet_log'].iloc[-1],
        'residual': 0.0
    }, name=next_date)
    history = pd.concat([history, placeholder.to_frame().T])

X_future = pd.DataFrame(future_rows).ffill().fillna(0.0)
dX_future = xgb.DMatrix(X_future[feature_cols])
pred_res_future = bst.predict(dX_future)

prophet_future_log = fc['yhat'].reindex(X_future.index).fillna(method='ffill').values
combined_log = prophet_future_log + pred_res_future
combined_price = np.exp(combined_log)

# Monte Carlo simulation
N_MC = 5000
residual_samples = y_train.values
mc_paths = np.exp(np.outer(np.ones(N_MC), prophet_future_log + pred_res_future) + np.random.choice(residual_samples, size=(N_MC,HORIZON_DAYS), replace=True))
mc_mean = mc_paths.mean(axis=0)
mc_low = np.percentile(mc_paths, 5, axis=0)
mc_high = np.percentile(mc_paths, 95, axis=0)

# Probabilities
last_close = df['close'].iloc[-1]
combined_delta = (combined_price - last_close)
combined_uncertainty = y_train.std() * last_close
prob_up = 1 / (1 + np.exp(-combined_delta / (combined_uncertainty + 1e-9)))

# Result dataframe
result_df = pd.DataFrame({
    'combined_price': combined_price,
    'mc_mean': mc_mean,
    'mc_low': mc_low,
    'mc_high': mc_high,
    'prob_up': prob_up
}, index=X_future.index)

# -------------------------
# 6) Directional backtest
# -------------------------
dX_test = xgb.DMatrix(X_test[feature_cols])
pred_res_test = bst.predict(dX_test)
prophet_test_log = fc['yhat'].reindex(X_test.index).fillna(method='ffill').values
pred_log_test = prophet_test_log + pred_res_test
pred_price_test = np.exp(pred_log_test)
true_price_test = df.loc[X_test.index,'close'].values

pred_dir = np.diff(pred_price_test) > 0
true_dir = np.diff(true_price_test) > 0
dir_acc = accuracy_score(true_dir, pred_dir)
print(f"Directional accuracy on holdout: {dir_acc:.4f}")

# -------------------------
# 7) Plot
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df['close'], label='Historical', color='blue')
plt.plot(result_df.index, combined_price, label='Hybrid (Prophet+XGB)', color='red')
plt.fill_between(result_df.index, mc_low, mc_high, color='purple', alpha=0.15, label='Monte Carlo 90% CI')
plt.legend()
plt.title(f"{TICKER} Hybrid Forecast")
plt.show()

# -------------------------
# 8) Save
# -------------------------
pd.set_option('display.float_format', lambda x: f"{x:,.2f}")
print("\n🔮 Combined forecast (first rows):")
print(result_df.head(15))

result_df.to_csv("nextgen_hybrid_forecast_results.csv")
print("\n✅ Forecast saved to nextgen_hybrid_forecast_results.csv")
