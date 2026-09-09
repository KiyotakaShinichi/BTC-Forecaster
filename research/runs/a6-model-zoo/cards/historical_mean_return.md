# historical_mean_return

**Family** BASELINE | **Status** ACTIVE | **Scientific status** EXPLORATORY

log(1 + mean training simple return).

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: nothing beyond the core install

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.001 s
- **Predict time**: 0.000 s
- **Parameters**: 1
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016481 |
| RMSE | 0.023396 |
| MASE (naive = 1) | 1.005278 |
| skill vs naive | -0.005278 |
| forecast bias | 0.001284 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.5000 |
| MCC | 0.0000 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 1.398, p = 0.1624, q = 0.2112 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.005469
- worst block -0.009695
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 207 bytes, sha256 `50796088c6ad267f`
- reload reproduces the forecasts bit-identically

## Notes

- Distinct from random_walk_drift by Jensen's inequality: the arithmetic mean of simple returns exceeds the mean log return by approximately half the return variance.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
