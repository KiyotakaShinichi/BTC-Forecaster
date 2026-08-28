# Market-intelligence local performance

Measured on 2026-08-28 using Python 3.14.3 and DuckDB 1.5.5 on the local Windows development environment. These are representative observations, not CI thresholds or cross-machine guarantees.

| Operation | Size | Wall time |
|---|---:|---:|
| SQL-filtered document query plus validation | 1,000 documents | 221.219 ms |
| SQL-filtered event query plus validation | 1,000 events | 209.815 ms |
| Immutable point-in-time snapshot | 1,000 documents + 1,000 events | 634.649 ms |
| Parquet replay dataset with persisted membership snapshots | 1,000 origins | 156,761.826 ms |

The initial replay implementation repeatedly queried and validated the entire evidence set for every origin and exceeded three minutes. B3 changed bulk replay to read bounded evidence once and batch snapshot integrity validation/insertion. Snapshot membership and point-in-time checks were not weakened.

Reproduce with:

```powershell
python -m market_intelligence.benchmark
```
