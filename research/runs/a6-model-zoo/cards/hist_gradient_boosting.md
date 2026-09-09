# hist_gradient_boosting

**Family** TREE_ENSEMBLE | **Status** ACTIVE | **Scientific status** EXPLORATORY

Histogram-binned gradient boosting, 300 iterations, early stopping off.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.452 s
- **Predict time**: 0.020 s
- **Parameters**: not reported
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.020171 |
| RMSE | 0.026796 |
| MASE (naive = 1) | 1.230408 |
| skill vs naive | -0.230408 |
| forecast bias | -0.010138 |
| directional accuracy | 0.5163 |
| balanced accuracy | 0.5158 |
| MCC | 0.0467 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 8.496, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.238427
- worst block -0.352441
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 313685 bytes, sha256 `261dd014eb01ea35`
- reload reproduces the forecasts bit-identically

## Notes

- Its advantage is speed at large n, which this lab does not have; it is here to check the binning approximation costs nothing at n=1,000.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
