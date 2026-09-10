# uc_stochastic_cycle

**Family** STATISTICAL | **Status** ACTIVE | **Scientific status** EXPLORATORY

Unobserved components: local level plus a stochastic cycle of estimated frequency.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 3.237 s
- **Predict time**: 0.176 s
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
| MAE | 0.016382 |
| RMSE | 0.023353 |
| MASE (naive = 1) | 0.999284 |
| skill vs naive | 0.000716 |
| forecast bias | -0.000396 |
| directional accuracy | 0.5149 |
| balanced accuracy | 0.5149 |
| MCC | 0.0298 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic -0.662, p = 0.5081, q = 0.5662 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks 0.000601
- worst block -0.000608
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 2422224 bytes, sha256 `04a029245b607315`
- reload reproduces the forecasts bit-identically

## Notes

- The honest way to ask 'is there a cycle' when no fixed seasonal period is justified: the frequency is estimated, not imposed.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
