# arima

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

ARIMA(1,0,1) with constant on log returns; d=0 because the series is already differenced.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.097 s
- **Predict time**: 0.041 s
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
| MAE | 0.016420 |
| RMSE | 0.023359 |
| MASE (naive = 1) | 1.001580 |
| skill vs naive | -0.001580 |
| forecast bias | 0.000651 |
| directional accuracy | 0.5035 |
| balanced accuracy | 0.5042 |
| MCC | 0.0342 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 0.623, p = 0.5334, q = 0.5778 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.001820
- worst block -0.004054
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 1452543 bytes, sha256 `0b7353ac437108e8`
- reload reproduces the forecasts bit-identically

## Notes

- Order chosen from {(1,0,0), (1,0,1), (2,0,2)} on DEV only.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
