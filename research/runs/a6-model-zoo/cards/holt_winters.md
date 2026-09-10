# holt_winters

**Family** STATISTICAL | **Status** UNSUITABLE_FOR_CONSTRAINED_LAB | **Scientific status** EXPLORATORY

Holt-Winters additive seasonal exponential smoothing at period 7.

## What it is

- **Capabilities declared**: none declared
- **Preprocessing**: declared per model
- **Requires**: statsmodels

## Training budget

- 1,000 deterministic rows, the contiguous tail of the training partition
- **Fit time**: 0.000 s
- **Predict time**: 0.000 s
- **Parameters**: not reported
- **Training-row fingerprint**: `n/a`

## What it produces

- point forecast of the next bar's log return
- **does not** produce quantiles (not declared)
- **does not** produce direction probability (not declared)
- **does not** produce variance (not declared)

## Why it did not run

```
{'reason': 'no weekly seasonality in the training rows: ACF(7) = -0.018 against a +/-0.062 white-noise band, day-of-week ANOVA p = 0.95, STL seasonal strength 0.065. A seasonal component here would fit noise, and at n=1,000 it would fit it confidently.'}
```

## Notes

- Registered rather than omitted: the measurement that excludes it is itself a result, and a missing row is indistinguishable from an oversight.

## Limitations

- Fitted on 1,000 training rows and scored on a single holdout block. That is a sample size at which almost nothing is decidable, and one partition is not a distribution.
- A6 is exploratory. A2's 36-fold walk-forward study remains the authoritative historical evidence, and a result here neither confirms nor overturns it -- the two are not comparable.
- No model in A6 is PROMOTED, and the paper-trading engine stays fail-closed. Nothing in this card is a recommendation to trade.
