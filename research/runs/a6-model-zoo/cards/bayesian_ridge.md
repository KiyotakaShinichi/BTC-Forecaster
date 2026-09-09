# bayesian_ridge

**Family** LINEAR_ML | **Status** ACTIVE | **Scientific status** EXPLORATORY

Ridge with the penalty inferred by evidence approximation.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: sklearn

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.005 s
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
| MAE | 0.016805 |
| RMSE | 0.023635 |
| MASE (naive = 1) | 1.025084 |
| skill vs naive | -0.025084 |
| forecast bias | -0.004060 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.4986 |
| MCC | -0.0376 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 2.844, p = 0.0046, q = 0.0069 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.026503
- worst block -0.050731
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 2732 bytes, sha256 `242071b067a2fd1b`
- reload reproduces the forecasts bit-identically

## Notes

- A control on how much the frozen alphas cost the other linear models.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
