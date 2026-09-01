"""B4 research run: build targets, run every declared study, write the registry.

Composes the B4 modules into one reproducible run. What it can actually study is
determined by the B4.0 audit, not by what would be interesting:

* Cross-asset lead/lag has real point-in-time-valid history and is run in full.
* Every intelligence-domain event study is declared and executed against the
  live store, which currently holds nothing -- so each returns an explicit
  INSUFFICIENT_DATA row rather than being skipped. An absent study that leaves
  no trace is indistinguishable from one that was never asked for.

The preregistration is written and hashed before the validation period is read.
The run manifest is written last.

Run:
    python scripts/b4_run_study.py --output research/market_intelligence/b4/runs
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from market_intelligence.b4.contracts import EvidenceTier, MarketSeries, SourceDomain
from market_intelligence.b4.crossasset import (
    CrossAssetFeatureKind,
    CrossAssetFeatureSpec,
    LeadLagResult,
    LeadLagSpec,
    compute_feature,
    forward_return,
    granger_study,
    lead_lag_study,
)
from market_intelligence.b4.eventstudy import (
    BenchmarkDefinition,
    EventFilter,
    EventStudyResult,
    EventStudySpec,
    OverlapPolicy,
    StudyEvent,
    equal_weighted_benchmark,
    run_event_study,
)
from market_intelligence.b4.market_data import (
    CachingMarketDataProvider,
    MarketDataProvider,
    YFinanceProvider,
    close_as_of,
)
from market_intelligence.b4.prereg import (
    CarryForwardPolicy,
    Preregistration,
    filter_to_period,
    split_history,
)
from market_intelligence.b4.registry import (
    SignalCandidate,
    SignalRegistry,
    decide,
    enforce_exploratory_budget,
    registry_summary,
)
from market_intelligence.b4.runner import RunRecorder, evidence_tier_counts, git_sha, make_run_id
from market_intelligence.b4.stability import (
    concentration,
    decay_profile,
    period_stability,
    summarise_stability,
)
from market_intelligence.b4.stats import (
    BootstrapConfig,
    CorrectedTest,
    Evidence,
    TestRecord,
    benjamini_hochberg,
    classify,
    threshold_from_dispersion,
)
from market_intelligence.b4.targets import (
    DEFAULT_HORIZONS_DAILY,
    TARGET_CONTRACT_VERSION,
    assert_targets_are_future_only,
    build_target_manifest,
    build_targets,
)

DAILY_SERIES: tuple[tuple[str, str, SourceDomain], ...] = (
    ("btc_usd_daily", "BTC-USD", SourceDomain.BTC_MARKET),
    ("dxy_daily", "DX-Y.NYB", SourceDomain.MACRO_MARKET),
    ("gold_daily", "GC=F", SourceDomain.CROSS_ASSET),
    ("oil_daily", "CL=F", SourceDomain.CROSS_ASSET),
    ("sp500_daily", "^GSPC", SourceDomain.CROSS_ASSET),
    ("nasdaq_daily", "^IXIC", SourceDomain.CROSS_ASSET),
    ("vix_daily", "^VIX", SourceDomain.CROSS_ASSET),
    ("ust10y_daily", "^TNX", SourceDomain.MACRO_MARKET),
    ("eth_usd_daily", "ETH-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE),
    ("bnb_usd_daily", "BNB-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE),
    ("xrp_usd_daily", "XRP-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE),
    ("ltc_usd_daily", "LTC-USD", SourceDomain.CRYPTO_MARKET_STRUCTURE),
)

CROSS_ASSET_IDS = (
    "dxy_daily",
    "gold_daily",
    "oil_daily",
    "sp500_daily",
    "nasdaq_daily",
    "vix_daily",
    "ust10y_daily",
)

# B4.22: the lag set is small and fixed. A 1d and a 5d window per asset, plus a
# volatility feature, across three horizons -- 45 declared cells, all reported.
LAG_WINDOWS = ("1d", "3d")
HORIZONS = ("1d", "3d", "7d")

# B4.5 / B4.6: the intelligence studies are declared here so that their absence
# is a recorded result rather than an omission.
DECLARED_EVENT_STUDIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("regulation", ("REGULATION",), ()),
    ("monetary_policy", ("MONETARY_POLICY",), ()),
    ("etf_flow", ("ETF_FLOW",), ()),
    ("institutional_adoption", ("INSTITUTIONAL_ADOPTION",), ()),
    ("exchange_incident", ("EXCHANGE_INCIDENT",), ()),
    ("security_incident", ("SECURITY_INCIDENT",), ()),
    ("macro_shock", ("MACRO_SHOCK",), ()),
    ("geopolitical_event", ("GEOPOLITICAL_EVENT",), ()),
    ("liquidity_event", ("LIQUIDITY_EVENT",), ()),
    ("entity_statement", ("ENTITY_STATEMENT",), ()),
    ("entity_trump", ("ENTITY_STATEMENT", "REGULATION"), ("Donald Trump",)),
    ("entity_musk", ("ENTITY_STATEMENT",), ("Elon Musk",)),
    ("entity_powell", ("MONETARY_POLICY", "ENTITY_STATEMENT"), ("Jerome Powell",)),
    ("entity_saylor", ("ENTITY_STATEMENT", "INSTITUTIONAL_ADOPTION"), ("Michael Saylor",)),
    ("entity_sec", ("REGULATION",), ("SEC",)),
)

# B4.7: contexts are studied separately and UNKNOWN is never pooled into either.
WHALE_STUDIES: tuple[tuple[str, str], ...] = (
    ("whale_exchange_inflow", "EXCHANGE_INFLOW"),
    ("whale_exchange_outflow", "EXCHANGE_OUTFLOW"),
    ("whale_custody", "CUSTODY"),
    ("whale_internal_transfer", "INTERNAL_TRANSFER"),
    ("whale_unknown_context", "UNKNOWN"),
)

# 1,000 replicates is ample for a percentile interval and keeps a 63-cell sweep
# over ~3,000 daily observations to minutes rather than hours. Declared here so
# it is part of the preregistration hash rather than a tuning knob.
BOOTSTRAP = BootstrapConfig(method="stationary", block_length=10, replicates=1000, seed=20260831)


def load_series(provider: MarketDataProvider) -> dict[str, MarketSeries]:
    series: dict[str, MarketSeries] = {}
    for series_id, ticker, domain in DAILY_SERIES:
        series[series_id] = provider.fetch(series_id, ticker, domain, "1d")
    return series


def daily_origins(start: datetime, end: datetime) -> list[datetime]:
    origins: list[datetime] = []
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end:
        origins.append(cursor)
        cursor += timedelta(days=1)
    return origins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("research/market_intelligence/b4/cache"))
    parser.add_argument("--output", type=Path, default=Path("research/market_intelligence/b4/runs"))
    parser.add_argument("--database", type=Path, default=None, help="intelligence store to study")
    parser.add_argument("--start", default="2018-01-01", help="earliest origin (BTC history is longer)")
    args = parser.parse_args()

    provider: MarketDataProvider = CachingMarketDataProvider(YFinanceProvider(), args.cache)
    print("loading market series ...", flush=True)
    series = load_series(provider)
    btc = series["btc_usd_daily"]

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = btc.end or datetime.now(timezone.utc)
    origins = [origin for origin in daily_origins(start, end) if close_as_of(btc, origin) is not None]
    print(f"  {len(series)} series, {len(origins)} daily origins {origins[0].date()} -> {origins[-1].date()}")

    run_id = make_run_id(
        btc.fingerprint(), str(len(origins)), TARGET_CONTRACT_VERSION, "b4-study-v1"
    )
    directory = args.output / run_id
    recorder = RunRecorder(directory, run_id=run_id)
    sha = git_sha()

    # ---------------------------------------------------------------- targets
    with recorder.stage("targets"):
        targets = build_targets(btc, origins, DEFAULT_HORIZONS_DAILY)
        assert_targets_are_future_only(btc, targets, DEFAULT_HORIZONS_DAILY)
        target_manifest = build_target_manifest(
            btc, targets, DEFAULT_HORIZONS_DAILY, git_sha=sha, created_at=datetime.now(timezone.utc)
        )
    recorder.write_json("target_manifest.json", target_manifest)
    recorder.write_json("targets.json", [row.as_record() for row in targets])
    print(f"  targets: {len(targets)} rows, fingerprint {target_manifest.series_fingerprint[:16]}")

    # -------------------------------------------------- preregistration split
    split = split_history(origins[0], origins[-1])
    development_rows = filter_to_period(
        # as_record() renders forecast_origin as an ISO string; the datetime has
        # to go in *after* the spread or the filter sees text.
        [{**row.as_record(), "forecast_origin": row.forecast_origin} for row in targets],
        split,
        period="development",
    )
    # Each horizon's threshold comes from *its own* development-period
    # dispersion. Deriving all three from the 1d return would set the 3d and 7d
    # bars far too low, because a 7-day return is several times as dispersed as
    # a one-day one.
    development_returns_by_horizon = {
        horizon: [
            value
            for row in development_rows
            if (value := row.get(f"forward_return_{horizon}")) is not None
        ]
        for horizon in HORIZONS
    }
    thresholds = tuple(
        threshold_from_dispersion(
            f"forward_return_{horizon}", development_returns_by_horizon[horizon], fraction=0.25
        )
        for horizon in HORIZONS
    )

    plan = Preregistration(
        created_at=datetime.now(timezone.utc),
        git_sha=sha,
        research_questions=(
            "Do regulatory, ETF, monetary-policy, entity-statement or whale events precede "
            "abnormal BTC returns?",
            "Do cross-asset moves (DXY, gold, oil, equities, VIX, yields) lead BTC?",
            "Does sentiment add information beyond event count?",
            "How quickly do external signals decay, and are effects stable across periods?",
        ),
        event_types=tuple(sorted({t for _, types, _ in DECLARED_EVENT_STUDIES for t in types})),
        entities=tuple(sorted({e for _, _, entities in DECLARED_EVENT_STUDIES for e in entities})),
        horizons=HORIZONS,
        extraction_quality_filters={"minimum_relevance": 0.5, "minimum_confidence": 0.5},
        minimum_event_count=30,
        bootstrap=BOOTSTRAP,
        practical_thresholds=thresholds,
        multiple_testing_families=("cross_asset_lead_lag", "event_studies", "whale_studies"),
        carry_forward_policy=CarryForwardPolicy(),
        period_split=split,
        predefined_composites=(
            "sentiment",
            "sentiment x relevance",
            "sentiment x relevance x confidence",
        ),
        notes=(
            "Composites are named before results are seen (B4.8). Thresholds are 0.25 x the "
            "development-period sd of the 1d forward return."
        ),
    )
    plan_path = plan.write(directory / "preregistration.json")
    recorder.result_hashes["preregistration.json"] = plan.content_hash()
    print(f"  preregistration {plan.content_hash()[:16]} -> {plan_path.name}")
    print(f"  split: {split.rationale}")

    # -------------------------------------------------- cross-asset lead/lag
    features = tuple(
        CrossAssetFeatureSpec(
            feature_id=f"{series_id}:{kind.value}:{window}",
            series_id=series_id,
            kind=kind,
            window=window,
        )
        for series_id in CROSS_ASSET_IDS
        for kind, window in (
            [(CrossAssetFeatureKind.LAGGED_RETURN, w) for w in LAG_WINDOWS]
            + [(CrossAssetFeatureKind.REALIZED_VOLATILITY, "7d")]
        )
    )
    lead_lag_spec = LeadLagSpec(
        study_id="cross_asset_lead_lag",
        features=features,
        horizons=HORIZONS,
        minimum_observations=200,
        bootstrap=BOOTSTRAP,
        family="cross_asset_lead_lag",
    )

    with recorder.stage("cross_asset"):
        lead_lag_results = lead_lag_study(lead_lag_spec, btc, series, origins)
    recorder.write_json("cross_asset_lead_lag.json", [result.model_dump(mode="json") for result in lead_lag_results])
    reported = sum(1 for result in lead_lag_results if not result.insufficient)
    print(f"  cross-asset: {len(lead_lag_results)} declared cells, {reported} estimable")

    # ------------------------------------------------------- crypto benchmark
    with recorder.stage("benchmark"):
        benchmark = equal_weighted_benchmark(
            [series[key] for key in ("eth_usd_daily", "bnb_usd_daily", "xrp_usd_daily", "ltc_usd_daily")],
            series_id="crypto_ex_btc",
        )
    print(f"  crypto ex-BTC benchmark: {len(benchmark)} common bars")

    # ---------------------------------------------------------- event studies
    events = load_events(args.database)
    event_specs: list[EventStudySpec] = []
    for study_id, event_types, entities in DECLARED_EVENT_STUDIES:
        event_specs.append(
            EventStudySpec(
                study_id=study_id,
                event_filter=EventFilter(
                    event_types=event_types,
                    entities=entities,
                    minimum_relevance=0.5,
                    minimum_confidence=0.5,
                    require_evidence_tier=EvidenceTier.PIT_VALIDATED,
                ),
                post_horizons=HORIZONS,
                benchmark=BenchmarkDefinition.CRYPTO_MARKET_EXCESS,
                minimum_event_count=30,
                bootstrap=BOOTSTRAP,
                multiple_testing_family="event_studies",
                overlap_policy=OverlapPolicy.AGGREGATE_CLUSTER,
                cluster_window_hours=24,
            )
        )
    for study_id, context in WHALE_STUDIES:
        event_specs.append(
            EventStudySpec(
                study_id=study_id,
                event_filter=EventFilter(
                    event_types=("WHALE_TRANSFER",),
                    transfer_contexts=(context,),
                    minimum_relevance=0.5,
                    minimum_confidence=0.5,
                ),
                post_horizons=HORIZONS,
                benchmark=BenchmarkDefinition.CRYPTO_MARKET_EXCESS,
                minimum_event_count=30,
                bootstrap=BOOTSTRAP,
                multiple_testing_family="whale_studies",
                overlap_policy=OverlapPolicy.AGGREGATE_CLUSTER,
                cluster_window_hours=24,
            )
        )

    with recorder.stage("event_studies"):
        event_results = [run_event_study(events, spec, btc, benchmark) for spec in event_specs]
    recorder.write_json("event_studies.json", [result.model_dump(mode="json") for result in event_results])
    estimable = sum(1 for result in event_results if not result.insufficient)
    print(f"  event studies: {len(event_results)} declared, {estimable} estimable, {len(events)} events in store")

    # ------------------------------------------- multiple testing and decision
    with recorder.stage("inference"):
        records: list[TestRecord] = []
        for cell in lead_lag_results:
            record = cell.as_test_record(lead_lag_spec.family)
            if record is not None:
                records.append(record)
        for spec, study in zip(event_specs, event_results, strict=True):
            for horizon, horizon_result in study.horizons.items():
                if horizon_result.bootstrap is None or horizon_result.descriptive.mean is None:
                    continue
                records.append(
                    TestRecord(
                        test_id=f"{spec.study_id}@{horizon}",
                        family=spec.multiple_testing_family,
                        n=horizon_result.descriptive.n,
                        effect=horizon_result.descriptive.mean,
                        lower=horizon_result.bootstrap.lower,
                        upper=horizon_result.bootstrap.upper,
                        p_value=horizon_result.bootstrap.p_value,
                    )
                )
        corrected = benjamini_hochberg(records, q_threshold=plan.carry_forward_policy.maximum_q_value)
    recorder.write_json("corrected_tests.json", [record.model_dump(mode="json") for record in corrected])
    print(f"  inference: {len(records)} tests across {len({r.family for r in records})} families")

    # ------------------------------------------------ stability, concentration
    by_test = {record.test_id: record for record in corrected}
    candidates: list[SignalCandidate] = []

    with recorder.stage("stability"):
        candidates.extend(
            _cross_asset_candidates(lead_lag_results, lead_lag_spec, by_test, plan, btc, series, origins)
        )
        candidates.extend(_event_candidates(event_specs, event_results, by_test, plan))

    candidates = enforce_exploratory_budget(candidates, plan.carry_forward_policy)
    registry = SignalRegistry(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        preregistration_hash=plan.content_hash(),
        candidates=tuple(candidates),
    )
    registry.write(directory / "signal_registry.json")
    recorder.result_hashes["signal_registry.json"] = _hash_of(directory / "signal_registry.json")

    summary = registry_summary(registry)
    print(f"  registry: {summary}")

    # ------------------------------------------------------------- decay, misc
    decay = [
        decay_profile(
            result.study_id,
            {
                horizon: [
                    value
                    for value in [result.horizons[horizon].descriptive.mean]
                    if value is not None
                ]
                for horizon in HORIZONS
            },
            HORIZONS,
        ).model_dump(mode="json")
        for result in event_results
    ]
    recorder.write_json("signal_decay.json", decay)

    with recorder.stage("granger"):
        granger_rows = []
        granger_records: list[TestRecord] = []
        btc_returns = _daily_returns(btc, origins)
        for series_id in CROSS_ASSET_IDS:
            predictor_returns = _daily_returns(series[series_id], origins)
            paired = min(len(predictor_returns), len(btc_returns))
            decision, result = granger_study(
                series_id,
                "btc_usd_daily",
                predictor_returns[:paired],
                btc_returns[:paired],
                max_lag=5,
                minimum_observations=200,
            )
            granger_rows.append(
                {
                    "predictor": series_id,
                    "gate": decision.model_dump(mode="json"),
                    "result": result.model_dump(mode="json") if result else None,
                }
            )
            if result is not None:
                granger_records.append(result.as_test_record("granger"))
        granger_corrected = benjamini_hochberg(granger_records, q_threshold=0.10) if granger_records else []
    recorder.write_json(
        "granger.json",
        {
            "tests": granger_rows,
            "corrected": [record.model_dump(mode="json") for record in granger_corrected],
        },
    )
    ran = sum(1 for row in granger_rows if row["result"] is not None)
    survivors = [record.test_id for record in granger_corrected if record.significant_at_q]
    print(f"  granger: {ran}/{len(granger_rows)} passed the gate, {len(survivors)} survive BH at q<=0.10")

    # ------------------------------------------------------------- manifest
    manifest = recorder.finish(
        source_git_sha=sha,
        intelligence_dataset_id=None,
        intelligence_row_count=0,
        target_dataset_fingerprint=target_manifest.series_fingerprint,
        target_contract_version=TARGET_CONTRACT_VERSION,
        preregistration_hash=plan.content_hash(),
        study_spec_hashes={spec.study_id: spec.spec_hash() for spec in event_specs},
        market_series_fingerprints={key: value.fingerprint() for key, value in series.items()},
        origin_count=len(origins),
        event_counts={spec.study_id: result.observation_count for spec, result in zip(event_specs, event_results, strict=True)},
        evidence_tier_counts=evidence_tier_counts([candidate.pit_status for candidate in candidates]),
        market_bar_count=sum(len(value) for value in series.values()),
        notes=(
            f"{len(events)} intelligence events available in the store",
            split.rationale,
        ),
    )
    print(f"\nrun {manifest.run_id} written to {directory}")
    print(f"manifest hash {manifest.content_hash()[:16]}")
    return 0


def _daily_returns(series: MarketSeries, origins: Sequence[datetime]) -> list[float]:
    """One-period returns on the origin grid, from closes available at each origin."""
    closes = [close_as_of(series, origin) for origin in origins]
    return [
        later / earlier - 1.0
        for earlier, later in zip(closes, closes[1:], strict=False)
        if earlier is not None and later is not None and earlier > 0.0
    ]


def load_events(database: Path | None) -> list[StudyEvent]:
    """Read study events from the intelligence store, if one exists.

    Returns an empty list when there is no store or no signals -- which is the
    expected outcome of this run and is treated as data, not as an error.
    """
    if database is None or not database.exists():
        return []
    from market_intelligence.storage import IntelligenceStore  # noqa: PLC0415

    store = IntelligenceStore(database)
    try:
        rows = store.connection.execute("SELECT payload FROM signals ORDER BY available_time").fetchall()
    finally:
        store.close()

    import json  # noqa: PLC0415

    events: list[StudyEvent] = []
    for (payload,) in rows:
        record = json.loads(payload)
        events.append(
            StudyEvent(
                event_id=record["event_id"],
                event_type=record["event_type"],
                entity=record.get("entity", "unknown"),
                event_time=record["event_time"],
                available_at=record["available_time"],
                btc_relevance=record.get("btc_relevance", 0.0),
                confidence=record.get("confidence", 0.0),
                novelty=record.get("novelty", 0.0),
                sentiment=record.get("sentiment", 0.0),
                transfer_context=record.get("transfer_context"),
                source_ids=tuple(record.get("source_ids", ())),
                provider=record.get("provider", "unknown"),
            )
        )
    return events


def _cross_asset_candidates(
    results: Sequence[LeadLagResult],
    spec: LeadLagSpec,
    by_test: Mapping[str, CorrectedTest],
    plan: Preregistration,
    btc: MarketSeries,
    series: Mapping[str, MarketSeries],
    origins: Sequence[datetime],
) -> list[SignalCandidate]:
    """One candidate per cross-asset feature, judged on its best horizon."""
    grouped: dict[str, list[LeadLagResult]] = {}
    for result in results:
        grouped.setdefault(result.feature_id, []).append(result)

    candidates: list[SignalCandidate] = []
    for feature_id, cells in grouped.items():
        effects = {cell.horizon: cell.slope_per_sd for cell in cells}
        intervals = {
            cell.horizon: ((cell.lower, cell.upper) if cell.lower is not None and cell.upper is not None else None)
            for cell in cells
        }
        tests = {
            cell.horizon: by_test.get(f"{feature_id}@{cell.horizon}") for cell in cells
        }
        q_values = {
            horizon: (test.q_value if test is not None else None) for horizon, test in tests.items()
        }

        evidence: dict[str, Evidence] = {}
        best: CorrectedTest | None = None
        for cell in cells:
            test = tests.get(cell.horizon)
            if test is None:
                evidence[cell.horizon] = Evidence.INSUFFICIENT_SAMPLE
                continue
            threshold = plan.threshold_for(f"forward_return_{cell.horizon}")
            verdict = classify(test, threshold, stable=None)
            evidence[cell.horizon] = verdict.evidence
            if best is None or test.q_value < best.q_value:
                best = test

        stability = None
        fragility = None
        if best is not None:
            horizon = best.test_id.rsplit("@", 1)[-1]
            paired = _paired_observations(spec, feature_id, horizon, btc, series, origins)
            if len(paired) >= 30:
                stability = period_stability([(moment, value) for moment, value in paired])
                fragility = concentration(
                    [(moment.date().isoformat(), value) for moment, value in paired], dimension="observation"
                )

        sample = max((cell.n for cell in cells), default=0)
        decision, reasons = decide(
            signal_id=feature_id,
            policy=plan.carry_forward_policy,
            sample_count=sample,
            effective_sample_count=sample,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=best,
            evidence=max(evidence.values(), default=None, key=_evidence_rank) if evidence else None,
            stable=summarise_stability(stability, fragility),
            fragile=fragility.fragile if fragility else None,
            largest_source_share=None,
        )
        candidates.append(
            SignalCandidate(
                signal_id=feature_id,
                definition=f"{spec.study_id}: {feature_id} against BTC forward return",
                source_domain="CROSS_ASSET",
                feature_version="b4-crossasset-v1",
                sample_count=sample,
                effective_sample_count=sample,
                tested_horizons=tuple(cell.horizon for cell in cells),
                effects=effects,
                intervals=intervals,
                q_values=q_values,
                evidence=evidence,
                stability_verdict=stability.verdict if stability else None,
                concentration_verdict=fragility.verdict if fragility else None,
                largest_source_share=None,
                pit_status=EvidenceTier.PIT_VALIDATED,
                multiple_testing_family=spec.family,
                decision=decision,
                decision_reasons=reasons,
            )
        )
    return candidates


def _paired_observations(
    spec: LeadLagSpec,
    feature_id: str,
    horizon: str,
    btc: MarketSeries,
    series: Mapping[str, MarketSeries],
    origins: Sequence[datetime],
) -> list[tuple[datetime, float]]:
    """Per-origin contribution used for stability and concentration checks."""
    feature = next((item for item in spec.features if item.feature_id == feature_id), None)
    if feature is None:
        return []
    source = series.get(feature.series_id)
    if source is None:
        return []
    paired: list[tuple[datetime, float]] = []
    for origin in origins:
        predictor = compute_feature(feature, source, origin)
        outcome = forward_return(btc, origin, horizon)
        if predictor is None or outcome is None:
            continue
        # Contribution of this origin to the standardised slope: sign-aligned
        # outcome, which is what a leave-one-out check needs to see.
        paired.append((origin, predictor * outcome))
    return paired


def _evidence_rank(evidence: Evidence) -> int:
    order = {
        Evidence.INSUFFICIENT_SAMPLE: 0,
        Evidence.NO_EVIDENCE: 1,
        Evidence.INCONCLUSIVE: 2,
        Evidence.WEAK: 3,
        Evidence.SUPPORTED: 4,
    }
    return order[evidence]


def _event_candidates(
    specs: Sequence[EventStudySpec],
    results: Sequence[EventStudyResult],
    by_test: Mapping[str, CorrectedTest],
    plan: Preregistration,
) -> list[SignalCandidate]:
    candidates: list[SignalCandidate] = []
    for spec, result in zip(specs, results, strict=True):
        effects = {
            horizon: value.descriptive.mean for horizon, value in result.horizons.items()
        }
        intervals = {
            horizon: (
                (value.bootstrap.lower, value.bootstrap.upper) if value.bootstrap else None
            )
            for horizon, value in result.horizons.items()
        }
        tests = {horizon: by_test.get(f"{spec.study_id}@{horizon}") for horizon in result.horizons}
        q_values = {
            horizon: (test.q_value if test is not None else None) for horizon, test in tests.items()
        }
        evidence = {
            horizon: (
                classify(test, plan.threshold_for(f"forward_return_{horizon}"), stable=None).evidence
                if test is not None
                else Evidence.INSUFFICIENT_SAMPLE
            )
            for horizon, test in tests.items()
        }
        best_candidates = [test for test in tests.values() if test is not None]
        best = min(best_candidates, key=lambda test: test.q_value) if best_candidates else None

        decision, reasons = decide(
            signal_id=spec.study_id,
            policy=plan.carry_forward_policy,
            sample_count=result.observation_count,
            effective_sample_count=result.effective_event_count,
            pit_status=EvidenceTier.PIT_VALIDATED if result.observation_count else None,
            best=best,
            evidence=max(evidence.values(), default=None, key=_evidence_rank) if evidence else None,
            stable=None,
            fragile=None,
            largest_source_share=None,
        )
        candidates.append(
            SignalCandidate(
                signal_id=spec.study_id,
                definition=(
                    f"event study: types={spec.event_filter.event_types} "
                    f"entities={spec.event_filter.entities or '(all)'} "
                    f"contexts={spec.event_filter.transfer_contexts or '(all)'}"
                ),
                source_domain="INTELLIGENCE",
                feature_version="b4-eventstudy-v1",
                sample_count=result.observation_count,
                effective_sample_count=result.effective_event_count,
                tested_horizons=tuple(result.horizons),
                effects=effects,
                intervals=intervals,
                q_values=q_values,
                evidence=evidence,
                stability_verdict=None,
                concentration_verdict=None,
                largest_source_share=None,
                pit_status=EvidenceTier.PIT_VALIDATED if result.observation_count else None,
                multiple_testing_family=spec.multiple_testing_family,
                decision=decision,
                decision_reasons=reasons,
                notes="; ".join(result.notes),
            )
        )
    return candidates


def _hash_of(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
