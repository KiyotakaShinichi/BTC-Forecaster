# rbf_svr

**Family** KERNEL_LOCAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

RBF-kernel SVR; the only non-linear member of the linear/kernel group.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.108 s
- **Predict time**: 0.105 s
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
| MAE | 0.021549 |
| RMSE | 0.028305 |
| MASE (naive = 1) | 1.314423 |
| skill vs naive | -0.314423 |
| forecast bias | -0.011822 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.4987 |
| MCC | -0.0039 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 10.399, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.306213
- worst block -0.375406
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: ljung_box, arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 101420 bytes, sha256 `79d3d1a956f21ce6`
- reload reproduces the forecasts bit-identically

## Notes

- Quadratic in the sample size, which is affordable only because n=1,000.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
