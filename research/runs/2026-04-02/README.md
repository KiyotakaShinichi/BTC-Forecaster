# Run artifacts — final pre-Track-A run (2026-04-02)

The last `bayesianCutoff.py` run before the Track A refactor, at commit
`952b304`. Kept as the before/after reference point.

Config (from `forecast_summary.json`): `HORIZON_DAYS=90`, `TEST_LAST_DAYS=30`,
`MONTE_CARLO_RUNS=300`, `BAYESIAN_TEMPERATURE=2.0`, MAP cutoff `2021-05-01`
selected from 48 candidates.

## Headline numbers, and why they are not what they appear

```
directional_accuracy        0.6897   p = 0.0307  ("significant")
walk_forward_mean_accuracy  0.5862 ± 0.0422 over 4 folds
```

**The 0.6897 / p=0.0307 holdout result is not a clean out-of-sample number.**
Three independent reasons, all confirmed against the code at `952b304`:

1. **Cutoff selected on the test set.** `quick_cutoff_eval` scored every
   candidate cutoff on the *last 90 days of the full series*, and the reported
   holdout is the last `TEST_LAST_DAYS` of that same series. The winning cutoff
   was chosen by looking at the data it was then scored on. With 48 candidates,
   a 0.69 maximum is close to what selection noise alone produces.
2. **The p-value is uncorrected and assumes independence.** One binomial test is
   reported, but 48 cutoffs were searched. Daily direction outcomes from an
   overlapping-feature model are also not independent trials.
3. **Directional accuracy measured the wrong difference.** `np.diff(pred_price)`
   compares the forecast path against itself rather than each prediction against
   the previous *actual* close, so the metric is not the tradeable quantity it
   was read as.

The walk-forward number (0.586) is the more honest of the two, and it is *also*
optimistic: the feature set and the cutoff were both selected using data that
overlaps every fold's test window.

These are the specific defects Track A was created to fix. See the leakage
section of the Track A report and `tests/test_leakage.py`.
