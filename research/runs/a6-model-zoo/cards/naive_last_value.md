# naive_last_value

**Family** BASELINE | **Status** ACTIVE | **Scientific status** EXPLORATORY

P[T+1] = P[T]; the forecast log return is exactly zero.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: nothing beyond the core install

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.001 s
- **Predict time**: 0.000 s
- **Parameters**: 0
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
| directional accuracy | 0.0000 |
| balanced accuracy | 0.5000 |
| MCC | 0.0000 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Stability

- mean skill across blocks 0.000000
- worst block 0.000000
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 201 bytes, sha256 `8c0fd3c0eaa27422`
- reload reproduces the forecasts bit-identically

## Notes

- Zero fitted parameters. Every conditional model in the zoo is measured against this one.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
