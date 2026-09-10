# local_level

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Stochastic local level on log price; the state-space form of simple exponential smoothing.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.488 s
- **Predict time**: 0.048 s
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
| MAE | 0.016394 |
| RMSE | 0.023364 |
| MASE (naive = 1) | 1.000000 |
| skill vs naive | 0.000000 |
| forecast bias | -0.000391 |
| directional accuracy | 0.4950 |
| balanced accuracy | 0.4950 |
| MCC | -0.0099 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic -0.470, p = 0.6386, q = 0.6554 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks 0.000000
- worst block -0.000000
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 731504 bytes, sha256 `4ad08f2dd7f81424`
- reload reproduces the forecasts bit-identically

## Notes

- Estimated by maximum likelihood over state variances, via the Kalman filter. Kalman is not registered as a separate model because it is the estimator for this family, not a model.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
