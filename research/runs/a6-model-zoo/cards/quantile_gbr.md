# quantile_gbr

**Family** PROBABILISTIC | **Status** ACTIVE | **Scientific status** EXPLORATORY

Gradient boosting on pinball loss, one model per quantile level.

## What it is

- **Capabilities declared**: DIRECTION_PROBABILITY, POINT, QUANTILES, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 26.789 s
- **Predict time**: 0.195 s
- **Parameters**: not reported
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- conditional quantiles (pinball loss reported)
- direction probability (Brier score and calibration reported)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.017852 |
| RMSE | 0.024588 |
| MASE (naive = 1) | 1.088912 |
| skill vs naive | -0.088912 |
| forecast bias | -0.007207 |
| directional accuracy | 0.5121 |
| balanced accuracy | 0.5115 |
| MCC | 0.0438 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 5.251, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.090228
- worst block -0.112252
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 1450027 bytes, sha256 `5f31c57552454bc4`
- reload reproduces the forecasts bit-identically

## Notes

- Estimates conditional quantiles. Not a confidence interval and it carries no coverage guarantee; calibration is measured, not claimed.
- Its point forecast is the conditional median, because that is what pinball loss at 0.5 estimates. It has no mean to report.
- Independently fitted quantiles can cross; the columns are sorted, which cannot hurt calibration because a crossed pair is a statement the model could not have meant.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
