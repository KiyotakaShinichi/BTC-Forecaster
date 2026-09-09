# tcn

**Family** DEEP | **Status** ACTIVE | **Scientific status** EXPLORATORY

Dilated residual causal convolutions at dilations 1, 2, 4.

## What it is

- **Capabilities declared**: POINT, SERIALIZE
- **Preprocessing**: declared per model
- **Requires**: nothing beyond the core install

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 9.873 s
- **Predict time**: 0.161 s
- **Parameters**: 2561
- **Training-row fingerprint**: `b7b0666da4280e44`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Measured

| metric | value |
|---|---|
| MAE | 0.023455 |
| RMSE | 0.031576 |
| MASE (naive = 1) | 1.430706 |
| skill vs naive | -0.430706 |
| forecast bias | -0.003533 |
| directional accuracy | 0.4993 |
| balanced accuracy | 0.4992 |
| MCC | -0.0016 |
| train-constant null | 0.4993 |
| beats that null | no |

The naive baseline's MAE on the same block is 0.016394.

## Against the baseline

- Diebold-Mariano statistic 11.439, p = 0.0000, q = 0.0000 after Benjamini-Hochberg
- **Verdict**: significantly **worse** than the naive baseline

## Stability

- mean skill across blocks -0.428524
- worst block -0.494109
- positive in every block: no

## Residual diagnostics

- n = 705
- assumptions rejected: ljung_box, arch_lm, jarque_bera, adf, kpss
- diagnostics describe how a model fails, not whether it is profitable

## Serialization

- 2115013 bytes, sha256 `855a8f0deaa94043`
- reload reproduces the forecasts bit-identically

## Notes

- Reaches 15 bars against cnn_1d's 5 at the same depth; that is the whole reason both are registered.
- Trained on the numpy autodiff engine in this repository, whose gradients are checked against central finite differences. torch is not a dependency: the lock is compiled with --all-extras and installed by the fresh-clone CI job, so a torch extra would put a multi-gigabyte CUDA closure in every clean install.
- Budget frozen and shared across all seven: lookback 24, max 60 epochs, batch 64, lr 0.003, patience 8, seed 20260909. Early stopping watches DEV.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
