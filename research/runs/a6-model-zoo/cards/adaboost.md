# adaboost

**Family** TREE_ENSEMBLE | **Status** ACTIVE | **Scientific status** EXPLORATORY

AdaBoost.R2 over depth-3 stumps, 200 rounds at lr=0.05.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 4.656 s
- **Predict time**: 0.061 s
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
| MAE | 0.017267 |
| RMSE | 0.024043 |
| MASE (naive = 1) | 1.053263 |
| skill vs naive | -0.053263 |
| forecast bias | -0.004184 |
| directional accuracy | 0.5035 |
| balanced accuracy | 0.5029 |
| MCC | 0.0109 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 4.523, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.054873
- worst block -0.074826
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 258988 bytes, sha256 `c84e2f5fc2a5653d`
- reload reproduces the forecasts bit-identically

## Notes

- Reweights toward hard examples, which on this series are largely noise -- a hypothesis worth testing rather than assuming.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
