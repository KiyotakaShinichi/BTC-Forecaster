# B5 Gate 1 — corpus sufficiency

**Decision: `INTELLIGENCE_CORPUS_INSUFFICIENT`.** Research evidence only; nothing here is a trading signal, and live trading is disabled.

- Audited as of `2026-09-11T00:00:00+00:00`, input `collector-state:default-root`
- Store sha256 `fe9ae0a03461ddb80acdb40aeeeaa385bab54eff4252b97aa624c69c90c44429`
- Content fingerprint `c298391baec5f19bd3e3bb0917f74a5a0ae206e0d0033706a57eb525b0de5a9c`
- Preregistration `70777f0e45b0b8ddd6b6d84b4a9502bb4277d5a0d2b4631c59816610a50fcc9d`
- Result digest `80702011c70b1494826bfa97ac9a5895532db3c0e8888359a4f9008489349689`

## Clauses

| clause | required | observed | met |
|---|---|---|---|
| `ready_families` | >= 1 family meeting every adequacy clause alone (30 events, 20 effective at 168h, 3 publishers, 1 provider, 180 days, 80% coverage) | 0 of 15 families ready | **no** |
| `point_in_time_integrity` | <= 5% of resolved, uncorrected events with impossible timestamps | 0.0% (0 of 4) | yes |
| `event_time_from_publication` | >= 90% of point-in-time-valid events aligned to a publication time | 100.0% (4 of 4) | yes |

## Funnel

| stage | events |
|---|---:|
| 1_collected | 7 |
| 2_not_invalidated | 4 |
| 3_sources_resolved | 4 |
| 4_point_in_time_valid | 4 |
| 5_quality | 0 |
| 6_publication_time | 0 |
| 7_independent_events | 0 |
| 8_effective_events | 0 |

## Checking it

`python -m market_intelligence.b5 verify <this directory>` recomputes every file's sha256 and the result digest. The store itself is not redistributed; its sha256 and the content fingerprint say whether a store someone else holds is the one audited here.
