# uc_stochastic_cycle

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Unobserved components: local level plus a stochastic cycle of estimated frequency.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 1.755 s
- **Predict time**: 0.053 s
- **Parameters**: 4
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.016451 |
| RMSE | 0.023373 |
| MASE (naive = 1) | 1.003494 |
| skill vs naive | -0.003494 |
| forecast bias | -0.000399 |
| directional accuracy | 0.4596 |
| balanced accuracy | 0.4596 |
| MCC | -0.0809 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 1.705, p = 0.0886, q = 0.1191 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.003538
- worst block -0.007093
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 5537840 bytes, sha256 `461e5fbb656963fe`
- reload reproduces the forecasts bit-identically

## Notes

- The honest way to ask 'is there a cycle' when no fixed seasonal period is justified: the frequency is estimated, not imposed.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
