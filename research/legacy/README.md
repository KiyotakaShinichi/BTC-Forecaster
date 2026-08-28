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

## bayesianCutoff.py

Moved here in the Track A refactor. It was the maintained implementation and the
one `api_server.py` invoked; `btc_forecaster/` replaces it, and the API now runs
`python -m btc_forecaster.cli run`.

Preserved rather than deleted because it is the reference for the before/after
comparison, and because the defects documented in the Track A report are all
verifiable against this exact file:

- Features and target on the same bar (`roll_mean_ret_*`, `ema_*`, `sma_*` at
  row D contain `close[D]`, and the target was the residual of `log_close[D]`).
- Cutoff chosen by scoring 48 candidates on the last 90 days, then the holdout
  reported on that same window.
- Feature selection performed once, outside the walk-forward loop whose folds
  it then contaminated.
- Monte Carlo shocks drawn independently per horizon step instead of
  accumulated along the path, so 95% bands did not widen with horizon.
- Directional accuracy computed as `np.diff` of the forecast against `np.diff`
  of the actual, comparing the forecast path with itself.
- A bare `except:` around cutoff evaluation that swallowed every failure,
  including KeyboardInterrupt.

Its final run output is frozen in `research/runs/2026-04-02/`.
