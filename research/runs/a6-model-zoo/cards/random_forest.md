# random_forest

**Family** TREE_ENSEMBLE | **Status** ACTIVE | **Scientific status** EXPLORATORY

300 bagged trees at the shared capacity ceiling.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 1.647 s
- **Predict time**: 0.129 s
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
| MAE | 0.017014 |
| RMSE | 0.023798 |
| MASE (naive = 1) | 1.037816 |
| skill vs naive | -0.037816 |
| forecast bias | -0.004887 |
| directional accuracy | 0.5078 |
| balanced accuracy | 0.5072 |
| MCC | 0.0314 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 3.363, p = 0.0008, q = 0.0013 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.038433
- worst block -0.048418
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 428015 bytes, sha256 `03e71d1fd856d57c`
- reload reproduces the forecasts bit-identically

## Notes

- Capacity frozen for n=1,000; an unbounded forest reaches training R^2 ~ 1 and evaluation skill ~ 0, and that gap is the lesson.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
