# holt_linear_trend

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Damped Holt linear trend on log price, innovations state-space form.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 2.936 s
- **Predict time**: 0.468 s
- **Parameters**: 5
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016386 |
| RMSE | 0.023343 |
| MASE (naive = 1) | 0.999536 |
| skill vs naive | 0.000464 |
| forecast bias | -0.000334 |
| directional accuracy | 0.4950 |
| balanced accuracy | 0.4951 |
| MCC | -0.0099 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic -0.172, p = 0.8632, q = 0.8836 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks 0.000119
- worst block -0.004340
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 5167143 bytes, sha256 `4841b4d479cc4a13`
- reload reproduces the forecasts bit-identically

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
