# Run b4-3980bac0b72aeaa0

Complete: `run_manifest.json` is present, which is this track's completion
marker. Every other file here is hashed in it.

## One artifact is deliberately withheld

`targets.json` — the 3,166 row-level BTC outcome records — is **not committed**.
It carries raw closing prices, and the data provider's terms do not grant
redistribution of raw quotes; only derived statistics are published. Everything
needed to audit it is here:

* `target_manifest.json` records the source series fingerprint, the timestamp
  convention, the horizon definitions, the row count, and the sha256 of the row
  set (`result_hash`).
* `run_manifest.json` records the same sha256 under `result_hashes`.

Regenerate and check it with:

```
python scripts/b4_run_study.py --output <a fresh directory>
```

then compare the new `targets.json` sha256 against `result_hashes` here. A match
confirms the published results were computed from exactly those outcomes.

`verify_run()` on this directory will therefore report `targets.json` as missing.
That is the intended state, not a corruption: the run is published minus one
file that cannot be redistributed, and the manifest names it and its hash.

## What is here

| file | what it holds |
| --- | --- |
| `preregistration.json` | the frozen analysis plan and its content hash |
| `cross_asset_lead_lag.json` | all 63 declared (feature, horizon) cells |
| `corrected_tests.json` | the same tests after Benjamini-Hochberg |
| `granger.json` | the 7 gate decisions and their bounded 5-lag tests |
| `event_studies.json` | all 20 declared intelligence studies, every one empty |
| `signal_decay.json` | decay profiles across horizons |
| `signal_registry.json` | 41 candidates and the clause that decided each |
| `target_manifest.json` | outcome dataset provenance |
| `run_manifest.json` | written last; hashes everything above |
