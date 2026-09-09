# xgboost

**Family** TREE_ENSEMBLE | **Status** ACTIVE | **Scientific status** EXPLORATORY

XGBoost on the causal matrix at the shared capacity ceiling.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: xgboost

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.416 s
- **Predict time**: 0.010 s
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
| MAE | 0.020108 |
| RMSE | 0.026691 |
| MASE (naive = 1) | 1.226512 |
| skill vs naive | -0.226512 |
| forecast bias | -0.010386 |
| directional accuracy | 0.5078 |
| balanced accuracy | 0.5072 |
| MCC | 0.0246 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 8.571, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.231645
- worst block -0.302244
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 369109 bytes, sha256 `f0a3aa8ff652e2c3`
- reload reproduces the forecasts bit-identically

## Notes

- NOT comparable to A2's xgboost_causal_retuned, which selected hyperparameters by nested inner validation inside all 36 folds. This one is frozen and fitted once on 1,000 rows.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
