# conformal_ridge

**Family** PROBABILISTIC | **Status** ACTIVE | **Scientific status** EXPLORATORY

Ridge with split-conformal intervals calibrated on the DEV block.

## What it is

- **Capabilities declared**: DIRECTION_PROBABILITY, POINT, QUANTILES, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.015 s
- **Predict time**: 0.013 s
- **Parameters**: 12
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- conditional quantiles (pinball loss reported)
- direction probability (Brier score and calibration reported)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.019022 |
| RMSE | 0.025482 |
| MASE (naive = 1) | 1.160283 |
| skill vs naive | -0.160283 |
| forecast bias | -0.010180 |
| directional accuracy | 0.5021 |
| balanced accuracy | 0.5014 |
| MCC | 0.0144 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 7.577, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.170930
- worst block -0.316255
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 1534 bytes, sha256 `53185d5d6aef9976`
- reload reproduces the forecasts bit-identically

## Notes

- The conformal coverage guarantee requires exchangeability, which a volatility-clustered daily return series does not satisfy. The nominal level is reported as nominal and the empirical coverage is measured; the gap between them is the result.
- Calibrated on DEV, never on HOLDOUT -- calibrating on the block it is about to be scored against would guarantee coverage by construction and measure nothing.
- Uses signed residual quantiles rather than absolute ones, so an asymmetric return distribution produces an asymmetric interval.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
