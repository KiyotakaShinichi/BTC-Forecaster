# knn

**Family** KERNEL_LOCAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Distance-weighted 25-NN on standardised causal features.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.013 s
- **Predict time**: 0.228 s
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
| MAE | 0.017149 |
| RMSE | 0.024073 |
| MASE (naive = 1) | 1.046066 |
| skill vs naive | -0.046066 |
| forecast bias | -0.003263 |
| directional accuracy | 0.5078 |
| balanced accuracy | 0.5074 |
| MCC | 0.0178 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 4.269, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.044699
- worst block -0.067856
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 118919 bytes, sha256 `e214031aa77e3314`
- reload reproduces the forecasts bit-identically

## Notes

- Meaningful because the features are approximately stationary -- lagged returns and rolling statistics, not raw prices -- so Euclidean distance is a defensible statement that two days looked alike.
- Limited by dimensionality: 11 features and 1,000 points is sparse, and 'nearest' begins to mean 'least far'. The distance to the k-th training neighbour is reported so that is visible as a number.
- Limited by temporal correlation: adjacent rows share rolling windows, so a query's neighbours are often its calendar neighbours, which inflates similarity without adding independent evidence.
- Fits nothing and stores everything, so it reports no parameter count rather than reporting the size of the training set.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
