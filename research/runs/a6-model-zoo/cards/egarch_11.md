# egarch_11

**Family** VOLATILITY | **Status** ACTIVE | **Scientific status** EXPLORATORY

EGARCH(1,1,1) with Student-t innovations: log-variance with a leverage term.

## What it is

- **Capabilities declared**: POINT, QUANTILES, SERIALIZE, VARIANCE
- **Preprocessing**: declared per model
- **Requires**: arch

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.226 s
- **Predict time**: 0.038 s
- **Parameters**: 6
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- conditional quantiles (pinball loss reported)
- conditional variance (QLIKE against the squared-return proxy)
- **does not** produce direction probability (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016436 |
| RMSE | 0.023374 |
| MASE (naive = 1) | 1.002548 |
| skill vs naive | -0.002548 |
| forecast bias | 0.000782 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.5000 |
| MCC | 0.0000 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 0.959, p = 0.3380, q = 0.3995 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.002649
- worst block -0.005516
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 1694 bytes, sha256 `dac1a1f50acb468b`
- reload reproduces the forecasts bit-identically

## Notes

- Forecasts conditional variance, not direction. Its POINT output is the fitted constant mean and will score like the naive baseline; that is the honest answer, not a failure.
- Quantiles come from the fitted Student-t conditional distribution, so they are the model's own statement rather than a Gaussian assumption added afterwards.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
