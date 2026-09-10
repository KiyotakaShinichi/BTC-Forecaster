# local_linear_trend

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Stochastic local linear trend on log price; the state-space form of Holt's method.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 2.369 s
- **Predict time**: 0.068 s
- **Parameters**: 3
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016445 |
| RMSE | 0.023371 |
| MASE (naive = 1) | 1.003109 |
| skill vs naive | -0.003109 |
| forecast bias | 0.000189 |
| directional accuracy | 0.5106 |
| balanced accuracy | 0.5108 |
| MCC | 0.0223 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 0.924, p = 0.3556, q = 0.4079 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.003383
- worst block -0.007200
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 3227302 bytes, sha256 `31e8c7cbde125684`
- reload reproduces the forecasts bit-identically

## Notes

- Shares its structure with holt_linear_trend and differs in estimation: variances by likelihood here, smoothing weights there. Two entries, one structure, different forecasts.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
