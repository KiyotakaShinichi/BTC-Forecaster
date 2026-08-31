"""B4.4 – B4.19 — the event-study engine, its inference, and its placebos.

With no historical intelligence corpus in existence, these tests are the only
evidence that the machinery works. So they are built around two synthetic worlds
with *known* answers:

* a world where events really are followed by a jump, which the engine must find
* a world where they are not, which the engine must refuse to find anything in

A framework that only passes the first is worthless -- it would confirm every
hypothesis B4 was asked to test. The second is the load-bearing one.

No network.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from market_intelligence.b4.contracts import (
    B4DataError,
    EvidenceTier,
    MarketBar,
    MarketSeries,
    SourceDomain,
    TimestampConvention,
)
from market_intelligence.b4.eventstudy import (
    BenchmarkDefinition,
    EventFilter,
    EventStudySpec,
    OverlapPolicy,
    StudyEvent,
    build_observations,
    cluster_events,
    effective_event_count,
    equal_weighted_benchmark,
    event_response,
    filter_events,
    pre_event_returns,
    pre_trend_warning,
    run_event_study,
)
from market_intelligence.b4.placebo import (
    matched_control_test,
    non_event_origins,
    permute_labels_within_blocks,
    placebo_test,
    realized_volatility_before,
    shift_events_within_blocks,
)
from market_intelligence.b4.stats import (
    BootstrapConfig,
    CorrectedTest,
    Evidence,
    PracticalThreshold,
    benjamini_hochberg,
    block_bootstrap,
    classify,
    describe,
    threshold_from_dispersion,
)
from market_intelligence.b4.stats import (
    TestRecord as StatTestRecord,
)

BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)
HOUR = 3_600
FAST_BOOTSTRAP = BootstrapConfig(replicates=300, block_length=6, seed=7)


# --------------------------------------------------------------- world builder


def price_series(closes: list[float], series_id: str = "btc") -> MarketSeries:
    return MarketSeries(
        series_id=series_id,
        ticker="BTC-USD",
        domain=SourceDomain.BTC_MARKET,
        provider="fixture",
        timestamp_convention=TimestampConvention.PERIOD_OPEN,
        evidence_tier=EvidenceTier.PIT_VALIDATED,
        source_timezone="UTC",
        bars=tuple(
            MarketBar(
                period_start=BASE + timedelta(hours=index),
                period_seconds=HOUR,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1.0,
            )
            for index, close in enumerate(closes)
        ),
    )


def random_walk(hours: int, *, seed: int, drift: float = 0.0, sigma: float = 0.004) -> list[float]:
    generator = random.Random(seed)
    level = 30_000.0
    closes = [level]
    for _ in range(hours - 1):
        level *= 1.0 + drift + generator.gauss(0.0, sigma)
        closes.append(level)
    return closes


def world_with_effect(
    event_hours: list[int], *, jump: float, hours: int = 2000, seed: int = 11
) -> tuple[MarketSeries, list[StudyEvent]]:
    """A world where each event is followed by a permanent `jump`.

    Bar k covers [k, k+1) and is available at k+1, so an event available at
    hour H is studied from a base close of bar H-1 -- struck entirely before the
    event was knowable -- and its first measurable response is bar H. The jump
    is therefore applied from index H onward. Getting this off by one bar makes
    the effect invisible, which is exactly what happened the first time.
    """
    closes = random_walk(hours, seed=seed)
    for event_hour in event_hours:
        for index in range(event_hour, hours):
            closes[index] *= 1.0 + jump
    return price_series(closes), _events_at(event_hours)


def world_without_effect(event_hours: list[int], *, hours: int = 2000, seed: int = 11) -> tuple[
    MarketSeries, list[StudyEvent]
]:
    """The same events, on a series that knows nothing about them."""
    return price_series(random_walk(hours, seed=seed)), _events_at(event_hours)


def _events_at(event_hours: list[int], entity: str = "SEC", event_type: str = "REGULATION") -> list[StudyEvent]:
    return [
        StudyEvent(
            event_id=f"e{index}",
            event_type=event_type,
            entity=entity,
            event_time=BASE + timedelta(hours=hour),
            available_at=BASE + timedelta(hours=hour),
            btc_relevance=0.9,
            confidence=0.9,
            novelty=0.9,
            sentiment=0.2,
            source_ids=(f"doc{index}",),
            provider="fixture-provider",
        )
        for index, hour in enumerate(event_hours)
    ]


def spec(**overrides: object) -> EventStudySpec:
    defaults: dict[str, object] = {
        "study_id": "test-study",
        "event_filter": EventFilter(event_types=("REGULATION",)),
        "post_horizons": ("1h", "6h", "24h"),
        "minimum_event_count": 10,
        "bootstrap": FAST_BOOTSTRAP,
        "overlap_policy": OverlapPolicy.KEEP_FIRST,
        "cluster_window_hours": 6,
    }
    defaults.update(overrides)
    return EventStudySpec(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ filtering


class TestEventFiltering:
    def test_thresholds_and_types_are_applied(self) -> None:
        events = [
            _events_at([10])[0],
            _events_at([20])[0].model_copy(update={"event_id": "low-rel", "btc_relevance": 0.1}),
            _events_at([30])[0].model_copy(update={"event_id": "other", "event_type": "ETF_FLOW"}),
        ]
        kept = filter_events(events, spec(event_filter=EventFilter(event_types=("REGULATION",), minimum_relevance=0.5)))
        assert [event.event_id for event in kept] == ["e0"]

    def test_retrospective_evidence_is_excluded_from_a_pit_study(self) -> None:
        """B4.36: the two tiers are never pooled."""
        events = [
            _events_at([10])[0],
            _events_at([20])[0].model_copy(
                update={"event_id": "retro", "evidence_tier": EvidenceTier.RETROSPECTIVE_ONLY}
            ),
        ]
        kept = filter_events(events, spec())
        assert [event.event_id for event in kept] == ["e0"]

    def test_an_event_without_provenance_is_excluded_when_required(self) -> None:
        events = [_events_at([10])[0].model_copy(update={"source_ids": ()})]
        assert filter_events(events, spec()) == []
        permissive = spec(event_filter=EventFilter(event_types=("REGULATION",), require_provenance=False))
        assert len(filter_events(events, permissive)) == 1

    def test_transfer_context_filters_do_not_pool_unknown(self) -> None:
        """B4.7: UNKNOWN is its own category, never folded into inflow."""
        events = [
            _events_at([10])[0].model_copy(update={"event_id": "in", "transfer_context": "EXCHANGE_INFLOW"}),
            _events_at([20])[0].model_copy(update={"event_id": "unk", "transfer_context": "UNKNOWN"}),
            _events_at([30])[0].model_copy(update={"event_id": "none", "transfer_context": None}),
        ]
        inflow_only = spec(
            event_filter=EventFilter(event_types=("REGULATION",), transfer_contexts=("EXCHANGE_INFLOW",))
        )
        assert [event.event_id for event in filter_events(events, inflow_only)] == ["in"]


# ----------------------------------------------------------------- clustering


class TestClustering:
    def test_forty_articles_about_one_event_are_one_cluster(self) -> None:
        """B4.9, stated as the number that would otherwise be wrong."""
        events = [
            _events_at([100])[0].model_copy(
                update={
                    "event_id": f"story{index}",
                    "available_at": BASE + timedelta(hours=100, minutes=3 * index),
                    "source_ids": (f"doc{index}",),
                }
            )
            for index in range(40)
        ]
        clusters = cluster_events(events, spec())
        assert len(clusters) == 1
        assert clusters[0].source_count == 40, "corroboration breadth is kept as metadata"
        assert len(clusters[0].events) == 40

    def test_events_beyond_the_window_are_separate_clusters(self) -> None:
        events = _events_at([100, 120])
        assert len(cluster_events(events, spec(cluster_window_hours=6))) == 2
        assert len(cluster_events(events, spec(cluster_window_hours=48))) == 1

    def test_clustering_chains_through_a_long_news_cycle(self) -> None:
        """Consecutive stories 4h apart over two days remain one cycle."""
        events = [
            _events_at([100])[0].model_copy(
                update={"event_id": f"s{index}", "available_at": BASE + timedelta(hours=100 + 4 * index)}
            )
            for index in range(12)
        ]
        assert len(cluster_events(events, spec(cluster_window_hours=6))) == 1

    def test_different_entities_never_share_a_cluster(self) -> None:
        events = [
            _events_at([100])[0],
            _events_at([100])[0].model_copy(update={"event_id": "musk", "entity": "Elon Musk"}),
        ]
        assert len(cluster_events(events, spec())) == 2

    def test_clustering_is_deterministic(self) -> None:
        events = _events_at([10, 12, 40, 41, 200])
        first = [cluster.cluster_id for cluster in cluster_events(events, spec())]
        shuffled = list(reversed(events))
        again = [cluster.cluster_id for cluster in cluster_events(shuffled, spec())]
        assert first == again


class TestOverlapPolicy:
    def test_the_policies_produce_different_sample_sizes(self) -> None:
        series, _ = world_without_effect([])
        events = [
            _events_at([100])[0].model_copy(
                update={"event_id": f"s{index}", "available_at": BASE + timedelta(hours=100, minutes=5 * index)}
            )
            for index in range(8)
        ]
        clusters_keep = cluster_events(events, spec(overlap_policy=OverlapPolicy.KEEP_FIRST))
        keep_first = build_observations(clusters_keep, spec(overlap_policy=OverlapPolicy.KEEP_FIRST), series)
        everything = build_observations(
            clusters_keep, spec(overlap_policy=OverlapPolicy.ALL_DEPENDENCE_AWARE), series
        )
        aggregated = build_observations(
            clusters_keep, spec(overlap_policy=OverlapPolicy.AGGREGATE_CLUSTER), series
        )
        assert len(keep_first) == 1
        assert len(everything) == 8
        assert len(aggregated) == 1

    def test_aggregate_uses_every_member_not_only_the_first(self) -> None:
        series = price_series([100.0 * (1.02**index) for index in range(400)])
        events = [
            _events_at([100])[0].model_copy(
                update={"event_id": f"s{index}", "available_at": BASE + timedelta(hours=100 + index)}
            )
            for index in range(3)
        ]
        study = spec(overlap_policy=OverlapPolicy.AGGREGATE_CLUSTER, cluster_window_hours=6)
        clusters = cluster_events(events, study)
        aggregated = build_observations(clusters, study, series)[0]
        first_only = build_observations(
            clusters, spec(overlap_policy=OverlapPolicy.KEEP_FIRST, cluster_window_hours=6), series
        )[0]
        assert aggregated.member_event_count == 3
        # On a compounding series the three members' 6h returns differ, so the
        # average is not the first member's value.
        assert aggregated.responses["6h"] != pytest.approx(first_only.responses["6h"], abs=0.0)

    def test_effective_count_falls_below_the_raw_count_when_windows_overlap(self) -> None:
        """B4.10: what the study is really powered on."""
        series, _ = world_without_effect([])
        events = [
            _events_at([100])[0].model_copy(
                update={"event_id": f"s{index}", "available_at": BASE + timedelta(hours=100 + index)}
            )
            for index in range(10)
        ]
        study = spec(overlap_policy=OverlapPolicy.ALL_DEPENDENCE_AWARE, cluster_window_hours=0)
        observations = build_observations(cluster_events(events, study), study, series)
        assert len(observations) == 10
        assert effective_event_count(observations, "24h") == 1
        assert effective_event_count(observations, "1h") == 10


# --------------------------------------------------------------- pre-event


class TestPreEventTrend:
    def test_pre_event_returns_look_backwards_only(self) -> None:
        series = price_series([100.0 + index for index in range(300)])
        origin = BASE + timedelta(hours=100)
        returns = pre_event_returns(series, origin, [-24, -6, -1])
        assert set(returns) == {"-24h", "-6h", "-1h"}
        assert all(value is not None and value > 0 for value in returns.values())

    def test_a_positive_offset_is_rejected(self) -> None:
        with pytest.raises(B4DataError, match="must be negative"):
            pre_event_returns(price_series([1.0, 2.0]), BASE, [6])

    def test_a_run_up_before_the_event_is_flagged_not_called_impact(self) -> None:
        """B4.11. The move happens *before* the events; the engine must say so."""
        hours = 1200
        closes = random_walk(hours, seed=3)
        event_hours = list(range(100, 1000, 40))
        for event_hour in event_hours:
            for index in range(event_hour - 6, hours):
                closes[index] *= 1.0 + 0.01  # the run-up starts before availability
        series = price_series(closes)
        result = run_event_study(_events_at(event_hours), spec(), series)
        warning = pre_trend_warning(result, "24h")
        assert warning is not None
        assert "reverse timing or leakage" in warning


# ------------------------------------------------------- the two known worlds


class TestTheEngineFindsARealEffect:
    """If this fails the engine is blind and every negative result is worthless."""

    def test_an_injected_jump_is_recovered_at_the_right_size(self) -> None:
        event_hours = list(range(100, 1900, 30))
        series, events = world_with_effect(event_hours, jump=0.02)
        result = run_event_study(events, spec(), series)

        assert result.observation_count == len(event_hours)
        assert not result.insufficient
        one_hour = result.horizons["1h"]
        assert one_hour.descriptive.mean is not None
        assert one_hour.descriptive.mean == pytest.approx(0.02, abs=0.004)
        assert one_hour.bootstrap is not None
        assert one_hour.bootstrap.lower > 0.0, "a real 2% jump must give an interval above zero"

    def test_the_placebo_does_not_reproduce_a_real_effect(self) -> None:
        event_hours = list(range(100, 1900, 30))
        series, events = world_with_effect(event_hours, jump=0.02)
        result = placebo_test(events, spec(), series, "1h", replicates=60, seed=5)
        assert result.exceedance_rate <= 0.05
        assert "larger than 95%" in result.verdict


class TestTheEngineFindsNothingWhenThereIsNothing:
    """The load-bearing test. A framework that cannot return a negative is a
    machine for confirming whatever it is pointed at."""

    def test_events_on_an_indifferent_series_give_an_interval_containing_zero(self) -> None:
        event_hours = list(range(100, 1900, 30))
        series, events = world_without_effect(event_hours)
        result = run_event_study(events, spec(), series)

        assert result.observation_count == len(event_hours)
        for horizon in ("1h", "6h", "24h"):
            bootstrap = result.horizons[horizon].bootstrap
            assert bootstrap is not None
            assert bootstrap.lower <= 0.0 <= bootstrap.upper, (
                f"{horizon}: a null world must not produce an interval excluding zero "
                f"(got [{bootstrap.lower:.5f}, {bootstrap.upper:.5f}])"
            )

    def test_the_placebo_reproduces_a_null_effect_routinely(self) -> None:
        event_hours = list(range(100, 1900, 30))
        series, events = world_without_effect(event_hours)
        result = placebo_test(events, spec(), series, "1h", replicates=60, seed=5)
        assert result.exceedance_rate > 0.05

    def test_a_small_sample_is_refused_rather_than_estimated(self) -> None:
        series, events = world_without_effect([100, 200, 300])
        result = run_event_study(events, spec(minimum_event_count=30), series)
        assert result.insufficient
        assert result.horizons["1h"].bootstrap is None
        assert any("below the declared minimum" in note for note in result.notes)

    def test_a_domain_with_no_events_still_produces_a_result_row(self) -> None:
        """An empty study must appear in the results table, not as a gap."""
        series, _ = world_without_effect([])
        result = run_event_study([], spec(), series)
        assert result.observation_count == 0
        assert result.insufficient
        assert result.origin_span is None
        assert result.horizons["1h"].descriptive.n == 0


# ------------------------------------------------------------------ benchmark


class TestBenchmark:
    def test_an_equal_weighted_index_tracks_its_constituents(self) -> None:
        first = price_series([100.0 * 1.01**index for index in range(50)], series_id="a")
        second = price_series([50.0 * 1.03**index for index in range(50)], series_id="b")
        index = equal_weighted_benchmark([first, second])
        # Each step grows by the average of 1% and 3%.
        growth = index.bars[1].close / index.bars[0].close
        assert growth == pytest.approx(1.02, abs=1e-9)

    def test_only_common_periods_are_used(self) -> None:
        first = price_series([100.0] * 50, series_id="a")
        short = price_series([100.0] * 20, series_id="b")
        assert len(equal_weighted_benchmark([first, short])) == 20

    def test_excess_return_removes_a_market_wide_move(self) -> None:
        market_move = [100.0 * 1.05**index for index in range(200)]
        series = price_series(market_move)
        benchmark = price_series(market_move, series_id="bench")
        excess = event_response(series, BASE + timedelta(hours=50), ["6h"], benchmark)
        assert excess["6h"] == pytest.approx(0.0, abs=1e-12)

    def test_a_raw_study_refuses_a_benchmark_and_vice_versa(self) -> None:
        series, events = world_without_effect([100, 200])
        with pytest.raises(B4DataError, match="must not be handed a benchmark"):
            run_event_study(events, spec(benchmark=BenchmarkDefinition.RAW), series, series)
        with pytest.raises(B4DataError, match="requires a benchmark series"):
            run_event_study(events, spec(benchmark=BenchmarkDefinition.CRYPTO_MARKET_EXCESS), series, None)

    def test_a_benchmark_needs_constituents_and_a_common_grid(self) -> None:
        with pytest.raises(B4DataError, match="at least one constituent"):
            equal_weighted_benchmark([])
        with pytest.raises(B4DataError, match="fewer than two common periods"):
            equal_weighted_benchmark([price_series([100.0], series_id="a")])


# ------------------------------------------------------------------ bootstrap


class TestBootstrap:
    def test_the_same_seed_gives_the_identical_interval(self) -> None:
        values = random_walk(400, seed=2)
        config = BootstrapConfig(replicates=200, block_length=8, seed=99)
        first = block_bootstrap(values, config)
        again = block_bootstrap(values, config)
        assert (first.lower, first.upper, first.p_value) == (again.lower, again.upper, again.p_value)

    def test_a_different_seed_moves_the_interval(self) -> None:
        values = random_walk(400, seed=2)
        first = block_bootstrap(values, BootstrapConfig(replicates=200, block_length=8, seed=1))
        other = block_bootstrap(values, BootstrapConfig(replicates=200, block_length=8, seed=2))
        assert (first.lower, first.upper) != (other.lower, other.upper)

    def test_longer_blocks_widen_the_interval_on_dependent_data(self) -> None:
        """The reason iid resampling is not used: it understates uncertainty.

        A strongly positively autocorrelated AR(1). Its long-run variance far
        exceeds its marginal variance, so resampling one observation at a time
        produces an interval that is far too narrow -- which is precisely the
        error a naive bootstrap would make on overlapping event windows.
        """
        generator = random.Random(23)
        dependent = [0.0] * 600
        for index in range(1, 600):
            dependent[index] = 0.95 * dependent[index - 1] + generator.gauss(0.0, 0.01)
        narrow = block_bootstrap(dependent, BootstrapConfig(replicates=600, block_length=1, seed=3))
        wide = block_bootstrap(dependent, BootstrapConfig(replicates=600, block_length=60, seed=3))
        assert (wide.upper - wide.lower) > 1.5 * (narrow.upper - narrow.lower)

    def test_both_methods_are_available_and_deterministic(self) -> None:
        values = random_walk(300, seed=4)
        for method in ("moving_block", "stationary"):
            config = BootstrapConfig(method=method, replicates=200, block_length=10, seed=8)  # type: ignore[arg-type]
            assert block_bootstrap(values, config).lower == block_bootstrap(values, config).lower

    def test_the_basic_interval_differs_from_the_percentile_interval(self) -> None:
        values = [float(index % 13) for index in range(200)]
        percentile = block_bootstrap(values, BootstrapConfig(replicates=200, seed=6, ci_method="percentile"))
        basic = block_bootstrap(values, BootstrapConfig(replicates=200, seed=6, ci_method="basic"))
        assert (percentile.lower, percentile.upper) != (basic.lower, basic.upper)

    def test_effective_sample_size_reflects_the_block_length(self) -> None:
        values = random_walk(200, seed=5)
        result = block_bootstrap(values, BootstrapConfig(replicates=150, block_length=20, seed=5))
        assert result.effective_sample_size == 200 // 20

    def test_a_single_observation_cannot_be_bootstrapped(self) -> None:
        with pytest.raises(B4DataError, match="at least two"):
            block_bootstrap([0.1], BootstrapConfig(replicates=100))


# ---------------------------------------------------------- multiple testing


class TestMultipleTesting:
    def test_benjamini_hochberg_matches_a_worked_example(self) -> None:
        """p = [.001, .008, .039, .041, .042], m = 5.

        Raw p*m/rank is [.005, .020, .065, .05125, .042]; the step-up pulls the
        middle two down to .042. Asserting the whole vector rather than one
        entry is deliberate -- the step-up is exactly the part that is easy to
        get wrong, and only the middle entries reveal it.
        """
        tests = [
            StatTestRecord(test_id=f"t{index}", family="f", n=100, effect=0.1, lower=0.0, upper=0.2, p_value=p)
            for index, p in enumerate([0.001, 0.008, 0.039, 0.041, 0.042])
        ]
        corrected = {record.test_id: record.q_value for record in benjamini_hochberg(tests, 0.05)}
        assert [corrected[f"t{index}"] for index in range(5)] == pytest.approx(
            [0.005, 0.020, 0.042, 0.042, 0.042]
        )

    def test_q_values_are_monotone_in_p(self) -> None:
        tests = [
            StatTestRecord(test_id=f"t{index}", family="f", n=50, effect=0.0, lower=-1.0, upper=1.0, p_value=p)
            for index, p in enumerate([0.01, 0.02, 0.03, 0.9])
        ]
        corrected = sorted(benjamini_hochberg(tests), key=lambda record: record.p_value)
        q_values = [record.q_value for record in corrected]
        assert q_values == sorted(q_values), "a larger p-value must never get a smaller q-value"

    def test_the_same_p_value_survives_alone_and_dies_in_a_crowd(self) -> None:
        """The same nominal result, corrected in two families of different size.

        This is the whole reason families are declared in advance: p=0.04 is a
        discovery on its own and is nothing once it is one of fifty tests whose
        other forty-nine are null.
        """
        alone = [StatTestRecord(test_id="a", family="small", n=50, effect=0.1, lower=0.0, upper=0.2, p_value=0.04)]
        crowd = [
            StatTestRecord(test_id="b0", family="large", n=50, effect=0.1, lower=0.0, upper=0.2, p_value=0.04)
        ] + [
            StatTestRecord(
                test_id=f"b{index}",
                family="large",
                n=50,
                effect=0.0,
                lower=-0.2,
                upper=0.2,
                p_value=0.5 + index / 200.0,
            )
            for index in range(1, 50)
        ]
        corrected = {record.test_id: record for record in benjamini_hochberg(alone + crowd, 0.05)}
        assert corrected["a"].q_value == pytest.approx(0.04)
        assert corrected["a"].significant_at_q
        # Raw q for b0 is 0.04*50/1 = 2.0, capped at 1.0, then pulled down by the
        # step-up to the largest member's q of 0.745*50/50.
        assert corrected["b0"].q_value == pytest.approx(0.745)
        assert not corrected["b0"].significant_at_q

    def test_identical_p_values_are_not_penalised_by_family_size(self) -> None:
        """BH is a false-discovery-rate procedure, not a Bonferroni penalty.

        When every test in a family shares a p-value the step-up is driven by
        the largest rank, so all of them pass or none do. Recorded because it
        looks like a bug the first time a results table shows it.
        """
        tests = [
            StatTestRecord(test_id=f"c{index}", family="f", n=50, effect=0.1, lower=0.0, upper=0.2, p_value=0.04)
            for index in range(50)
        ]
        corrected = benjamini_hochberg(tests, 0.05)
        assert all(record.q_value == pytest.approx(0.04) for record in corrected)
        assert all(record.significant_at_q for record in corrected)

    def test_a_family_of_pure_noise_yields_no_discoveries(self) -> None:
        """The multiple-testing guarantee, stated as the failure it prevents."""
        generator = random.Random(17)
        tests = [
            StatTestRecord(
                test_id=f"n{index}",
                family="noise",
                n=200,
                effect=0.0,
                lower=-1.0,
                upper=1.0,
                p_value=generator.random(),
            )
            for index in range(200)
        ]
        corrected = benjamini_hochberg(tests, 0.10)
        discoveries = [record for record in corrected if record.significant_at_q]
        uncorrected = [record for record in corrected if record.p_value <= 0.05]
        assert len(uncorrected) >= 5, "roughly one in twenty passes uncorrected, as expected"
        assert discoveries == [], "and none of them survives correction"

    def test_an_invalid_threshold_is_rejected(self) -> None:
        with pytest.raises(B4DataError, match="q_threshold"):
            benjamini_hochberg([], 1.5)


# --------------------------------------------------- practical significance


def corrected(**overrides: object) -> CorrectedTest:
    payload: dict[str, object] = {
        "test_id": "t",
        "family": "f",
        "n": 100,
        "effect": 0.02,
        "lower": 0.01,
        "upper": 0.03,
        "p_value": 0.001,
        "q_value": 0.01,
        "family_size": 10,
        "significant_at_q": True,
    }
    payload.update(overrides)
    return CorrectedTest(**payload)  # type: ignore[arg-type]


THRESHOLD = PracticalThreshold(
    outcome="forward_return_1h",
    minimum_absolute_effect=0.01,
    justification="test",
    minimum_sample=30,
    q_threshold=0.10,
)


class TestPracticalSignificance:
    def test_a_large_precise_stable_corrected_effect_is_supported(self) -> None:
        assert classify(corrected(), THRESHOLD, stable=True).evidence is Evidence.SUPPORTED

    def test_an_underpowered_test_is_never_supported_however_significant(self) -> None:
        """Checked before anything else, on purpose."""
        verdict = classify(corrected(n=5, q_value=0.0001), THRESHOLD, stable=True)
        assert verdict.evidence is Evidence.INSUFFICIENT_SAMPLE

    def test_failing_correction_downgrades_to_weak(self) -> None:
        assert classify(corrected(q_value=0.4), THRESHOLD).evidence is Evidence.WEAK

    def test_instability_downgrades_to_weak_not_to_no_evidence(self) -> None:
        verdict = classify(corrected(), THRESHOLD, stable=False)
        assert verdict.evidence is Evidence.WEAK
        assert any("leaving out one event" in reason for reason in verdict.reasons)

    def test_an_interval_containing_zero_is_inconclusive(self) -> None:
        assert classify(corrected(lower=-0.01, upper=0.05), THRESHOLD).evidence is Evidence.INCONCLUSIVE

    def test_a_precisely_estimated_tiny_effect_is_no_evidence(self) -> None:
        verdict = classify(corrected(effect=0.0001, lower=0.00005, upper=0.00015), THRESHOLD)
        assert verdict.evidence is Evidence.NO_EVIDENCE
        assert any("too small to matter" in reason for reason in verdict.reasons)

    def test_a_threshold_is_derived_from_dispersion_not_chosen(self) -> None:
        values = random_walk(300, seed=9)
        returns = [later / earlier - 1.0 for earlier, later in zip(values, values[1:], strict=False)]
        threshold = threshold_from_dispersion("forward_return_1h", returns, fraction=0.25)
        stats = describe(returns)
        assert stats.std is not None
        assert threshold.minimum_absolute_effect == pytest.approx(0.25 * stats.std)
        assert "development-period standard deviation" in threshold.justification

    def test_a_threshold_cannot_be_derived_from_one_observation(self) -> None:
        with pytest.raises(B4DataError, match="fewer than two"):
            threshold_from_dispersion("x", [0.1])


# -------------------------------------------------------------------- placebo


class TestPlaceboMechanics:
    def test_shifting_preserves_the_event_count_and_moves_the_times(self) -> None:
        events = _events_at([100, 200, 300])
        shifted = shift_events_within_blocks(events, block_hours=48, generator=random.Random(1))
        assert len(shifted) == len(events)
        assert [event.available_at for event in shifted] != [event.available_at for event in events]

    def test_shifting_keeps_event_time_and_availability_in_step(self) -> None:
        events = _events_at([100])
        shifted = shift_events_within_blocks(events, block_hours=10, generator=random.Random(2))
        original = events[0]
        assert shifted[0].available_at - shifted[0].event_time == original.available_at - original.event_time

    def test_permuting_preserves_timestamps_and_the_label_multiset(self) -> None:
        events = _events_at([10, 11, 12, 13]) + _events_at([14, 15], entity="Elon Musk", event_type="ENTITY_STATEMENT")
        permuted = permute_labels_within_blocks(events, block_hours=48, generator=random.Random(3))
        assert sorted(event.available_at for event in permuted) == sorted(
            event.available_at for event in events
        )
        assert sorted((event.event_type, event.entity) for event in permuted) == sorted(
            (event.event_type, event.entity) for event in events
        )

    def test_an_unknown_placebo_method_is_rejected(self) -> None:
        series, events = world_without_effect([100, 200])
        with pytest.raises(B4DataError, match="unknown placebo method"):
            placebo_test(events, spec(), series, "1h", method="magic")

    def test_a_non_positive_block_is_rejected(self) -> None:
        with pytest.raises(B4DataError, match="must be positive"):
            shift_events_within_blocks(_events_at([1]), block_hours=0, generator=random.Random(1))


class TestMatchedControls:
    def test_controls_near_an_event_are_excluded(self) -> None:
        series, _ = world_without_effect([])
        event_origins = [BASE + timedelta(hours=500)]
        candidates = non_event_origins(series, event_origins, step_hours=24, exclusion_hours=72)
        assert all(abs(candidate - event_origins[0]) > timedelta(hours=72) for candidate in candidates)

    def test_matching_pairs_events_with_comparable_controls(self) -> None:
        series, _ = world_without_effect([])
        event_origins = [BASE + timedelta(hours=hour) for hour in range(200, 1200, 100)]
        candidates = non_event_origins(series, event_origins, step_hours=12)
        result = matched_control_test(event_origins, candidates, series, "6h", seed=4)
        assert result.matched_pairs > 0
        assert result.event_stats.n == result.control_stats.n

    def test_a_null_world_shows_no_difference_from_controls(self) -> None:
        series, _ = world_without_effect([])
        event_origins = [BASE + timedelta(hours=hour) for hour in range(200, 1600, 60)]
        candidates = non_event_origins(series, event_origins, step_hours=12)
        result = matched_control_test(event_origins, candidates, series, "6h", seed=4)
        assert result.verdict.startswith("event origins behave like")

    def test_matching_requires_events_and_uncontaminated_candidates(self) -> None:
        series, _ = world_without_effect([])
        with pytest.raises(B4DataError, match="at least one event origin"):
            matched_control_test([], [BASE], series, "1h")
        with pytest.raises(B4DataError, match="exclusion window"):
            matched_control_test([BASE + timedelta(hours=100)], [BASE + timedelta(hours=101)], series, "1h")

    def test_pre_event_volatility_is_backward_looking(self) -> None:
        series, _ = world_without_effect([])
        origin = BASE + timedelta(hours=500)
        assert realized_volatility_before(series, origin, 24) is not None
        assert realized_volatility_before(series, BASE, 24) is None
