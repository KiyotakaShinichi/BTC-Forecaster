# pls

**Family** LINEAR_ML | **Status** ACTIVE | **Scientific status** EXPLORATORY

PLS regression with 2 components.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.010 s
- **Predict time**: 0.001 s
- **Parameters**: 12
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.018816 |
| RMSE | 0.025340 |
| MASE (naive = 1) | 1.147739 |
| skill vs naive | -0.147739 |
| forecast bias | -0.009885 |
| directional accuracy | 0.5021 |
| balanced accuracy | 0.5014 |
| MCC | 0.0377 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 7.193, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.156635
- worst block -0.285931
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 34640 bytes, sha256 `93726ac21861dc69`
- reload reproduces the forecasts bit-identically

## Notes

- Eleven features from one price series do not carry eleven directions.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
