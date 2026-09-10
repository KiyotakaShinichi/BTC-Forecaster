# arima

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

ARIMA(1,0,1) with constant on log returns; d=0 because the series is already differenced.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 1.828 s
- **Predict time**: 0.160 s
- **Parameters**: 4
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016439 |
| RMSE | 0.023368 |
| MASE (naive = 1) | 1.002729 |
| skill vs naive | -0.002729 |
| forecast bias | 0.001074 |
| directional accuracy | 0.5064 |
| balanced accuracy | 0.5070 |
| MCC | 0.0381 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 0.725, p = 0.4684, q = 0.5074 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.003284
- worst block -0.007905
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 3304140 bytes, sha256 `acbeda0216f84f54`
- reload reproduces the forecasts bit-identically

## Notes

- Order chosen from {(1,0,0), (1,0,1), (2,0,2)} on DEV only.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
