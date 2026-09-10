# random_walk_drift

**Family** BASELINE | **Status** ACTIVE | **Scientific status** EXPLORATORY

Constant drift: the mean training log return.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: nothing beyond the core install

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.002 s
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
| MAE | 0.016425 |
| RMSE | 0.023369 |
| MASE (naive = 1) | 1.001909 |
| skill vs naive | -0.001909 |
| forecast bias | 0.000642 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.5000 |
| MCC | 0.0000 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 0.814, p = 0.4159, q = 0.4634 after Benjamini-Hochberg
- **Verdict**: not distinguishable from the naive baseline

## Stability

- mean skill across blocks -0.001993
- worst block -0.004355
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 202 bytes, sha256 `012d3566d0eba424`
- reload reproduces the forecasts bit-identically

## Notes

- Mathematically identical to the classical endpoint drift (log P_end - log P_start) / (n - 1), because log returns telescope. Registered once rather than twice.
- A2's strongest model over 36 folds, at MAE skill +0.003957 -- which was still a REJECT.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
