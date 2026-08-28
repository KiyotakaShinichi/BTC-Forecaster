# Run artifacts — initial commit (2025-03-13)

Frozen output from the scripts now in `research/legacy/`, as committed in
`1cb1396` and previously scattered across the repository root.

| File | Produced by |
| --- | --- |
| `bayesian_ensemble_forecast.png` | cutoff-ensemble experiment |
| `hybrid_forecast_montecarlo.png`, `hybrid_forecast_results.csv` | `MC_Automation.py` lineage |
| `nextgen_hybrid_forecast_results.csv` | pre-Monte-Carlo hybrid forecast |
| `nextgen_hybrid_forecast_results_montecarlo.csv` | hybrid + GARCH Monte Carlo bands |
| `learning_curve.png`, `pacf_diagnostic.png` | diagnostics |

No manifest exists for these: the scripts predate provenance tracking, so the
exact data vintage that produced them is unrecoverable. Treat them as
illustrative history, not as reproducible results.
