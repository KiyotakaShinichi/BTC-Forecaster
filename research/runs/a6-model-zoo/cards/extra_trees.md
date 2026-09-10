# extra_trees

**Family** TREE_ENSEMBLE | **Status** ACTIVE | **Scientific status** EXPLORATORY

300 extremely randomised trees at the shared ceiling.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.376 s
- **Predict time**: 0.148 s
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
| MAE | 0.016765 |
| RMSE | 0.023622 |
| MASE (naive = 1) | 1.022628 |
| skill vs naive | -0.022628 |
| forecast bias | -0.004035 |
| directional accuracy | 0.5092 |
| balanced accuracy | 0.5086 |
| MCC | 0.0472 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 2.641, p = 0.0085, q = 0.0118 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.022689
- worst block -0.028481
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 522633 bytes, sha256 `3acb57907c6ab285`
- reload reproduces the forecasts bit-identically

## Notes

- Random split thresholds: more variance reduction, more bias.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
