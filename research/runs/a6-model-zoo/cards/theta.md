# theta

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Theta method (theta=2): frozen SES level plus half the fitted drift.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.041 s
- **Predict time**: 0.003 s
- **Parameters**: 2
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016915 |
| RMSE | 0.023856 |
| MASE (naive = 1) | 1.031770 |
| skill vs naive | -0.031770 |
| forecast bias | -0.000127 |
| directional accuracy | 0.5092 |
| balanced accuracy | 0.5093 |
| MCC | 0.0185 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 2.873, p = 0.0042, q = 0.0065 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.030872
- worst block -0.042973
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 236 bytes, sha256 `62d7d270795e4f07`
- reload reproduces the forecasts bit-identically

## Notes

- The forward recursion is written out because ThetaModel has no apply(refit=False), and refitting at every origin would be a different protocol -- one whose parameters have seen the holdout.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
