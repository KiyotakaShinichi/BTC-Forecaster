# decision_tree

**Family** TREE_ENSEMBLE | **Status** ACTIVE | **Scientific status** EXPLORATORY

Single regression tree, max_depth=4, min_samples_leaf=20.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.023 s
- **Predict time**: 0.002 s
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
| MAE | 0.016847 |
| RMSE | 0.023648 |
| MASE (naive = 1) | 1.027625 |
| skill vs naive | -0.027625 |
| forecast bias | -0.000173 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.4999 |
| MCC | -0.0003 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 2.690, p = 0.0073, q = 0.0106 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.026341
- worst block -0.050972
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 3029 bytes, sha256 `832008f792a2a95d`
- reload reproduces the forecasts bit-identically

## Notes

- The interpretable control the ensembles are measured against.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
