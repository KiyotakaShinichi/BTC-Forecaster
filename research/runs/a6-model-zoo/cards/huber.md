# huber

**Family** LINEAR_ML | **Status** ACTIVE | **Scientific status** EXPLORATORY

Huber loss: squared inside epsilon, absolute outside.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.075 s
- **Predict time**: 0.001 s
- **Parameters**: 12
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.017839 |
| RMSE | 0.024454 |
| MASE (naive = 1) | 1.088132 |
| skill vs naive | -0.088132 |
| forecast bias | -0.007448 |
| directional accuracy | 0.5007 |
| balanced accuracy | 0.5000 |
| MCC | 0.0002 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 5.517, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.094897
- worst block -0.187621
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 2424 bytes, sha256 `892cf91139d1b772`
- reload reproduces the forecasts bit-identically

## Notes

- Included specifically because daily crypto returns carry real outliers.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
