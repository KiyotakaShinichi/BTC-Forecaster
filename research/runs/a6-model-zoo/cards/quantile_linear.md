# quantile_linear

**Family** PROBABILISTIC | **Status** ACTIVE | **Scientific status** EXPLORATORY

Koenker-Bassett linear quantile regression, one fit per level.

## What it is

- **Capabilities declared**: DIRECTION_PROBABILITY, POINT, QUANTILES, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 1.542 s
- **Predict time**: 0.016 s
- **Parameters**: 84
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- conditional quantiles (pinball loss reported)
- direction probability (Brier score and calibration reported)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.017181 |
| RMSE | 0.023895 |
| MASE (naive = 1) | 1.047986 |
| skill vs naive | -0.047986 |
| forecast bias | -0.005335 |
| directional accuracy | 0.5035 |
| balanced accuracy | 0.5029 |
| MCC | 0.0269 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 4.126, p = 0.0000, q = 0.0001 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.051332
- worst block -0.098547
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 876883 bytes, sha256 `d07c97e3947f5e05`
- reload reproduces the forecasts bit-identically

## Notes

- The linear control for quantile_gbr: same loss, same levels, no capacity to fit a non-linear conditional distribution.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
