# Legacy implementations (not maintained)

These are the superseded ancestors of the forecasting pipeline. They are kept
because each one records a research decision that is not written down anywhere
else. They are **not** imported by `btc_forecaster/`, are excluded from the test
suite, and should not be run expecting current behaviour.

Lineage, oldest first (reconstructed from each file's own docstring header):

| File | Original name | What it added |
| --- | --- | --- |
| `BTC_v1.py` | `..._forecast_v3.py` | Prophet + XGBoost residual hybrid; PACF lag selection; Optuna tuning (100 trials); binomial significance test; learning curve. |
| `BTC_Predictor.py` | `..._forecast_v4.py` | Optuna objective over `TimeSeriesSplit`; rolling/EMA/SMA trend features; Monte Carlo uncertainty. |
| `predefined_optuna.py` | `..._forecast_v5.py` | Froze the Optuna search result into hardcoded XGBoost hyperparameters; added PACF visualisation and extended holdout. |
| `MC_Automation.py` | `..._montecarlo.py` | GARCH(1,1) on residuals feeding a Monte Carlo band; automatic rolling/EMA/SMA window selection. |
| `cutoffOptimization.py` | `..._WITH_AUTO_CUTOFF.py` | Replaced the hardcoded `START = "2021-01-01"` with a search over candidate cutoff dates scored on directional accuracy + MAE. |

`bayesianCutoff.py` (kept at the repository root) is the direct successor to
`cutoffOptimization.py`: it replaced argmax cutoff selection with a
softmax/temperature posterior over cutoffs and added regime detection and
walk-forward evaluation. It was the only script maintained after the initial
commit and is the one the API invokes.

## Why this matters for the current code

The XGBoost hyperparameters still used by the platform:

```python
{"max_depth": 3, "learning_rate": 0.010349570637285655,
 "subsample": 0.8021272578985711, "colsample_bytree": 0.7728798862419759}
```

are not arbitrary — they are the frozen output of the Optuna search in
`BTC_v1.py` / `BTC_Predictor.py`. Deleting these files would have made those
constants unexplainable. They are also **stale**: they were tuned against a
different cutoff, a different feature set, and a search that used
`TimeSeriesSplit` without an embargo. Re-tuning them under the walk-forward
engine is tracked as open quant debt.
