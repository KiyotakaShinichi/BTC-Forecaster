# nbeats

**Family** DEEP | **Status** ACTIVE | **Scientific status** EXPLORATORY

Two generic residual backcast/forecast blocks.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: nothing beyond the core install

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 1.279 s
- **Predict time**: 0.020 s
- **Parameters**: 18034
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.020919 |
| RMSE | 0.028173 |
| MASE (naive = 1) | 1.275993 |
| skill vs naive | -0.275993 |
| forecast bias | -0.006027 |
| directional accuracy | 0.5149 |
| balanced accuracy | 0.5147 |
| MCC | 0.0306 |
| train-constant null | 0.4993 |
| beats that null | yes |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 8.713, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.282823
- worst block -0.373518
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: ljung_box, arch_lm, jarque_bera, adf
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 2363095 bytes, sha256 `89b379d1c5ecb996`
- reload reproduces the forecasts bit-identically

## Notes

- Generic basis rather than trend/seasonality: a seasonal basis would impose a weekly period this series does not have.
- The parameter-heavy outlier by construction: its backcast head projects back to the full flattened window, giving roughly 18 parameters per training sequence against 1-9 for the other six. Left that way on purpose -- the ratio is the finding.
- Trained on the numpy autodiff engine in this repository, whose gradients are checked against central finite differences. torch is not a dependency: the lock is compiled with --all-extras and installed by the fresh-clone CI job, so a torch extra would put a multi-gigabyte CUDA closure in every clean install.
- Budget frozen and shared across all seven: lookback 24, max 60 epochs, batch 64, lr 0.003, patience 8, seed 20260909. Early stopping watches DEV.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
