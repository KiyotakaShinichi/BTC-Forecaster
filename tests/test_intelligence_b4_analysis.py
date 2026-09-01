"""B4.20 – B4.42 — cross-asset lead/lag, stability, preregistration, registry.

Same discipline as the event-study tests: synthetic worlds with known answers,
including a world where the answer is "nothing". No network.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.b4.contracts import (
    B4DataError,
    EvidenceTier,
    MarketBar,
    MarketSeries,
    SourceDomain,
    TimestampConvention,
)
from market_intelligence.b4.crossasset import (
    CrossAssetFeatureKind,
    CrossAssetFeatureSpec,
    LeadLagSpec,
    compute_feature,
    granger_gate,
    lead_lag_study,
)
from market_intelligence.b4.prereg import (
    CarryForwardPolicy,
    Preregistration,
    filter_to_period,
    require_preregistration,
    split_history,
)
from market_intelligence.b4.registry import (
    SignalCandidate,
    SignalDecision,
    SignalRegistry,
    decide,
    enforce_exploratory_budget,
    load_registry,
    registry_summary,
)
from market_intelligence.b4.stability import (
    concentration,
    decay_profile,
    period_stability,
    stratify_by_regime,
    summarise_stability,
)
from market_intelligence.b4.stats import (
    BootstrapConfig,
    CorrectedTest,
    Evidence,
    PracticalThreshold,
)

BASE = datetime(2022, 1, 1, tzinfo=timezone.utc)
DAY = 86_400
FAST = BootstrapConfig(replicates=200, block_length=5, seed=13)


def daily_series(closes: list[float], series_id: str, ticker: str = "X") -> MarketSeries:
    return MarketSeries(
        series_id=series_id,
        ticker=ticker,
        domain=SourceDomain.CROSS_ASSET,
        provider="fixture",
        timestamp_convention=TimestampConvention.PERIOD_OPEN,
        evidence_tier=EvidenceTier.PIT_VALIDATED,
        source_timezone="UTC",
        bars=tuple(
            MarketBar(
                period_start=BASE + timedelta(days=index),
                period_seconds=DAY,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1.0,
            )
            for index, close in enumerate(closes)
        ),
    )


def daily_origins(count: int, offset_days: int = 5) -> list[datetime]:
    return [BASE + timedelta(days=offset_days + index) for index in range(count)]


# ------------------------------------------------------------------ features


class TestCrossAssetFeatures:
    def test_a_lagged_return_uses_only_closes_available_at_the_origin(self) -> None:
        series = daily_series([100.0 * 1.01**index for index in range(60)], "asset")
        origin = BASE + timedelta(days=30)
        spec = CrossAssetFeatureSpec(
            feature_id="f", series_id="asset", kind=CrossAssetFeatureKind.LAGGED_RETURN, window="1d"
        )
        value = compute_feature(spec, series, origin)
        # Bar k is available at k+1, so at day 30 the last available close is
        # bar 29 and the one before it is bar 28.
        assert value == pytest.approx(series.bars[29].close / series.bars[28].close - 1.0)

    def test_realized_volatility_is_zero_on_a_constant_series(self) -> None:
        series = daily_series([100.0] * 60, "flat")
        spec = CrossAssetFeatureSpec(
            feature_id="f", series_id="flat", kind=CrossAssetFeatureKind.REALIZED_VOLATILITY, window="7d"
        )
        assert compute_feature(spec, series, BASE + timedelta(days=30)) == pytest.approx(0.0)

    def test_relative_strength_needs_a_reference(self) -> None:
        series = daily_series([100.0 + index for index in range(60)], "asset")
        spec = CrossAssetFeatureSpec(
            feature_id="f", series_id="asset", kind=CrossAssetFeatureKind.RELATIVE_STRENGTH, window="3d"
        )
        with pytest.raises(B4DataError, match="needs a reference series"):
            compute_feature(spec, series, BASE + timedelta(days=30))
        reference = daily_series([100.0] * 60, "ref")
        assert compute_feature(spec, series, BASE + timedelta(days=30), reference) is not None

    def test_a_zscore_needs_enough_history_and_some_variation(self) -> None:
        spec = CrossAssetFeatureSpec(
            feature_id="f",
            series_id="asset",
            kind=CrossAssetFeatureKind.ZSCORED_MOVE,
            window="1d",
            normalisation_window="30d",
        )
        flat = daily_series([100.0] * 60, "asset")
        assert compute_feature(spec, flat, BASE + timedelta(days=40)) is None
        generator = random.Random(1)
        noisy = daily_series([100.0 * (1.0 + generator.gauss(0, 0.02)) for _ in range(60)], "asset")
        assert compute_feature(spec, noisy, BASE + timedelta(days=40)) is not None

    def test_a_feature_before_the_series_starts_is_none(self) -> None:
        series = daily_series([100.0] * 10, "asset")
        spec = CrossAssetFeatureSpec(
            feature_id="f", series_id="asset", kind=CrossAssetFeatureKind.LAGGED_RETURN, window="7d"
        )
        assert compute_feature(spec, series, BASE) is None


# ------------------------------------------------------------------ lead/lag


class TestLeadLag:
    def _spec(self, **overrides: object) -> LeadLagSpec:
        payload: dict[str, object] = {
            "study_id": "leadlag",
            "features": (
                CrossAssetFeatureSpec(
                    feature_id="asset_1d",
                    series_id="asset",
                    kind=CrossAssetFeatureKind.LAGGED_RETURN,
                    window="1d",
                ),
            ),
            "horizons": ("1d", "3d"),
            "minimum_observations": 50,
            "bootstrap": FAST,
        }
        payload.update(overrides)
        return LeadLagSpec(**payload)  # type: ignore[arg-type]

    def test_an_injected_lead_relationship_is_recovered(self) -> None:
        """BTC tomorrow is a scaled copy of the asset's move today."""
        generator = random.Random(5)
        asset_returns = [generator.gauss(0.0, 0.02) for _ in range(400)]
        asset = [100.0]
        btc = [30_000.0]
        for index, move in enumerate(asset_returns):
            asset.append(asset[-1] * (1.0 + move))
            # BTC's next move copies the asset's *previous* move, so the
            # relationship is a genuine lead, not a contemporaneous one.
            lead = asset_returns[index - 1] if index else 0.0
            btc.append(btc[-1] * (1.0 + 0.8 * lead))

        results = lead_lag_study(
            self._spec(),
            daily_series(btc, "btc"),
            {"asset": daily_series(asset, "asset"), "btc": daily_series(btc, "btc")},
            daily_origins(350, offset_days=10),
        )
        one_day = next(result for result in results if result.horizon == "1d")
        assert one_day.slope_per_sd is not None
        assert one_day.slope_per_sd > 0.0
        assert one_day.lower is not None and one_day.lower > 0.0
        assert one_day.correlation is not None and one_day.correlation > 0.5

    def test_independent_series_give_an_interval_containing_zero(self) -> None:
        generator = random.Random(6)
        asset = [100.0]
        btc = [30_000.0]
        for _ in range(400):
            asset.append(asset[-1] * (1.0 + generator.gauss(0.0, 0.02)))
            btc.append(btc[-1] * (1.0 + generator.gauss(0.0, 0.03)))

        results = lead_lag_study(
            self._spec(),
            daily_series(btc, "btc"),
            {"asset": daily_series(asset, "asset")},
            daily_origins(350, offset_days=10),
        )
        for result in results:
            assert result.lower is not None and result.upper is not None
            assert result.lower <= 0.0 <= result.upper, (
                f"{result.feature_id}@{result.horizon}: independent series must not "
                f"produce an interval excluding zero"
            )

    def test_every_declared_cell_is_reported_even_when_underpowered(self) -> None:
        """B4.22/B4.12: no cell is dropped for looking uninteresting."""
        series = daily_series([100.0 + index for index in range(40)], "asset")
        results = lead_lag_study(
            self._spec(minimum_observations=1000),
            daily_series([30_000.0 + index for index in range(40)], "btc"),
            {"asset": series},
            daily_origins(20, offset_days=10),
        )
        assert len(results) == 2
        assert all(result.insufficient for result in results)
        assert all("below the declared minimum" in result.note for result in results)
        assert all(result.as_test_record("f") is None for result in results)

    def test_a_missing_series_is_an_error_not_a_silent_skip(self) -> None:
        with pytest.raises(B4DataError, match="no series"):
            lead_lag_study(self._spec(), daily_series([1.0, 2.0], "btc"), {}, daily_origins(5))

    def test_a_constant_predictor_is_reported_as_uninformative(self) -> None:
        flat = daily_series([100.0] * 400, "asset")
        generator = random.Random(7)
        btc = [30_000.0]
        for _ in range(400):
            btc.append(btc[-1] * (1.0 + generator.gauss(0.0, 0.02)))
        results = lead_lag_study(
            self._spec(),
            daily_series(btc, "btc"),
            {"asset": flat},
            daily_origins(350, offset_days=10),
        )
        assert all(result.insufficient for result in results)
        assert any("no variation" in result.note for result in results)


class TestGrangerGate:
    def test_a_short_sample_is_refused(self) -> None:
        decision = granger_gate([0.1] * 50, [0.2] * 50, minimum_observations=200)
        assert not decision.run
        assert "below the 200" in decision.reason

    def test_a_non_stationary_series_is_refused(self) -> None:
        trending = [float(index) for index in range(400)]
        noise = [random.Random(3).gauss(0, 1) for _ in range(400)]
        decision = granger_gate(trending, noise)
        assert not decision.run
        assert "stationarity" in decision.reason
        assert decision.predictor_stationary is False

    def test_two_stationary_series_of_adequate_length_pass_the_gate(self) -> None:
        generator = random.Random(4)
        first = [generator.gauss(0, 1) for _ in range(400)]
        second = [generator.gauss(0, 1) for _ in range(400)]
        decision = granger_gate(first, second)
        assert decision.run
        assert decision.stationarity_checked

    def test_too_few_observations_per_lag_is_refused(self) -> None:
        generator = random.Random(4)
        values = [generator.gauss(0, 1) for _ in range(210)]
        decision = granger_gate(values, values, max_lag=30, minimum_observations=200)
        assert not decision.run
        assert "ten per estimated lag" in decision.reason


# ------------------------------------------------------------------ stability


class TestDecay:
    def test_a_decaying_signal_is_described_as_monotone(self) -> None:
        profile = decay_profile(
            "s",
            {"1h": [0.04] * 10, "6h": [0.02] * 10, "24h": [0.01] * 10},
            ["1h", "6h", "24h"],
        )
        assert profile.monotone_decreasing
        assert profile.peak_horizon == "1h"
        assert profile.points[1].ratio_to_first == pytest.approx(0.5)

    def test_a_signal_peaking_later_is_not_forced_into_a_decay_curve(self) -> None:
        profile = decay_profile(
            "s",
            {"1h": [0.005] * 10, "6h": [0.03] * 10, "24h": [0.01] * 10},
            ["1h", "6h", "24h"],
        )
        assert not profile.monotone_decreasing
        assert profile.peak_horizon == "6h"
        assert "not monotone" in profile.note

    def test_an_empty_profile_says_so(self) -> None:
        profile = decay_profile("s", {}, ["1h", "6h"])
        assert profile.peak_horizon is None
        assert "no horizon" in profile.note


class TestPeriodStability:
    def _observations(self, per_year: dict[int, list[float]]) -> list[tuple[datetime, float]]:
        return [
            (datetime(year, 6, 1, tzinfo=timezone.utc) + timedelta(hours=index), value)
            for year, values in per_year.items()
            for index, value in enumerate(values)
        ]

    def test_a_consistent_effect_across_years_is_stable(self) -> None:
        result = period_stability(self._observations({2022: [0.01] * 30, 2023: [0.011] * 30}))
        assert result.stable
        assert result.sign_consistent
        assert result.verdict.startswith("STABLE")

    def test_a_sign_flip_between_periods_is_unstable(self) -> None:
        result = period_stability(self._observations({2022: [0.02] * 30, 2023: [-0.02] * 30}))
        assert not result.stable
        assert "changes sign" in result.verdict

    def test_one_era_carrying_the_whole_effect_is_unstable(self) -> None:
        """B4.29 stated as the failure it exists to catch."""
        result = period_stability(self._observations({2022: [0.0001] * 40, 2023: [0.05] * 40}))
        assert not result.stable
        assert "dropping one period" in result.verdict
        assert result.max_relative_shift is not None and result.max_relative_shift > 0.5

    def test_a_single_period_cannot_be_assessed(self) -> None:
        result = period_stability(self._observations({2022: [0.01] * 40}))
        assert not result.stable
        assert result.verdict.startswith("INSUFFICIENT")

    def test_thin_periods_are_reported_but_do_not_drive_the_verdict(self) -> None:
        result = period_stability(
            self._observations({2021: [5.0, -5.0], 2022: [0.01] * 30, 2023: [0.011] * 30})
        )
        assert any(period.label == "2021" for period in result.periods)
        assert result.stable, "a two-observation year must not make a stable signal look fragile"

    def test_no_observations_is_an_error(self) -> None:
        with pytest.raises(B4DataError, match="at least one observation"):
            period_stability([])


class TestConcentration:
    def test_one_outlier_carrying_the_result_is_fragile(self) -> None:
        """B4.28. Nine tiny values and one huge one."""
        values = [("e", 0.0005)] * 9 + [("big", 0.5)]
        result = concentration(values, dimension="event")
        assert result.fragile
        assert result.verdict.startswith("FRAGILE")
        assert result.top_contributors[0] == ("big", 0.5)

    def test_a_broad_result_is_not_fragile(self) -> None:
        values = [(f"e{index}", 0.01 + (index % 3) * 0.001) for index in range(60)]
        result = concentration(values, dimension="event")
        assert not result.fragile
        assert result.verdict.startswith("DIVERSE")

    def test_one_entity_supplying_most_events_is_reported(self) -> None:
        """B4.30. 'ENTITY_STATEMENT looks useful' when 90% is one person."""
        values = [("Elon Musk", 0.01 + index * 1e-5) for index in range(90)] + [
            (f"other{index}", 0.01) for index in range(10)
        ]
        result = concentration(values, dimension="entity")
        assert result.largest_share == pytest.approx(0.9)
        assert result.largest_share_key == "Elon Musk"
        assert result.verdict.startswith("CONCENTRATED")

    def test_a_single_publisher_dominating_is_reported(self) -> None:
        """B4.31, the same machinery pointed at sources."""
        values = [("one-publisher", 0.02) for _ in range(80)] + [
            (f"p{index}", 0.02) for index in range(20)
        ]
        result = concentration(values, dimension="source")
        assert result.largest_share == pytest.approx(0.8)
        assert "one-publisher" in result.verdict

    def test_no_observations_is_an_error(self) -> None:
        with pytest.raises(B4DataError, match="at least one observation"):
            concentration([], dimension="event")


class TestRegimeStratification:
    def test_thin_cells_are_refused(self) -> None:
        result = stratify_by_regime(
            [("high_vol", 0.01)] * 5 + [("low_vol", 0.02)] * 5, dimension="volatility"
        )
        assert not result.adequate
        assert result.verdict.startswith("INSUFFICIENT FOR REGIME ANALYSIS")

    def test_adequate_cells_are_stratified(self) -> None:
        result = stratify_by_regime(
            [("high_vol", 0.01)] * 40 + [("low_vol", 0.02)] * 40, dimension="volatility"
        )
        assert result.adequate
        assert len(result.cells) == 2


class TestStabilitySummary:
    def test_unassessed_is_none_not_false(self) -> None:
        """Returning False for an unassessed signal would demote every small study."""
        assert summarise_stability(None, None) is None

    def test_an_unstable_result_is_false(self) -> None:
        unstable = period_stability(
            [
                (datetime(2022, 6, 1, tzinfo=timezone.utc) + timedelta(hours=index), 0.02)
                for index in range(30)
            ]
            + [
                (datetime(2023, 6, 1, tzinfo=timezone.utc) + timedelta(hours=index), -0.02)
                for index in range(30)
            ]
        )
        assert summarise_stability(unstable, None) is False

    def test_a_fragile_concentration_is_false(self) -> None:
        fragile = concentration([("e", 0.0005)] * 9 + [("big", 0.5)], dimension="event")
        assert summarise_stability(None, fragile) is False


# ------------------------------------------------------------ preregistration


def a_plan(**overrides: object) -> Preregistration:
    payload: dict[str, object] = {
        "created_at": datetime(2026, 8, 31, tzinfo=timezone.utc),
        "git_sha": "abc123",
        "research_questions": ("do regulatory events precede abnormal returns?",),
        "event_types": ("REGULATION",),
        "entities": ("SEC",),
        "horizons": ("1h", "24h"),
        "extraction_quality_filters": {"minimum_relevance": 0.5},
        "minimum_event_count": 30,
        "bootstrap": FAST,
        "practical_thresholds": (
            PracticalThreshold(
                outcome="forward_return_1h", minimum_absolute_effect=0.004, justification="0.25 sd"
            ),
        ),
        "multiple_testing_families": ("event_studies",),
        "carry_forward_policy": CarryForwardPolicy(),
        "period_split": split_history(BASE, BASE + timedelta(days=1000)),
    }
    payload.update(overrides)
    return Preregistration(**payload)  # type: ignore[arg-type]


class TestPreregistration:
    def test_the_hash_covers_every_declared_choice(self) -> None:
        first = a_plan()
        assert first.content_hash() == a_plan().content_hash()
        assert first.content_hash() != a_plan(horizons=("1h", "24h", "72h")).content_hash()
        assert (
            first.content_hash()
            != a_plan(carry_forward_policy=CarryForwardPolicy(maximum_q_value=0.2)).content_hash()
        )

    def test_a_plan_round_trips_through_disk(self, tmp_path: Path) -> None:
        plan = a_plan()
        path = plan.write(tmp_path / "prereg.json")
        assert Preregistration.read(path).content_hash() == plan.content_hash()

    def test_rewriting_the_same_plan_is_a_no_op(self, tmp_path: Path) -> None:
        plan = a_plan()
        plan.write(tmp_path / "prereg.json")
        assert plan.write(tmp_path / "prereg.json").exists()

    def test_overwriting_with_a_different_plan_is_refused(self, tmp_path: Path) -> None:
        """B4.42. A preregistration that can be edited is not one."""
        a_plan().write(tmp_path / "prereg.json")
        with pytest.raises(B4DataError, match="already holds a different preregistration"):
            a_plan(horizons=("6h",)).write(tmp_path / "prereg.json")

    def test_editing_the_file_after_freezing_is_detected(self, tmp_path: Path) -> None:
        path = a_plan().write(tmp_path / "prereg.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["minimum_event_count"] = 5
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(B4DataError, match="edited after it was frozen"):
            Preregistration.read(path)

    def test_a_final_period_analysis_without_a_plan_is_refused(self) -> None:
        with pytest.raises(B4DataError, match="requires a preregistration"):
            require_preregistration(None, None)

    def test_a_mismatched_hash_reference_is_refused(self) -> None:
        with pytest.raises(B4DataError, match="different preregistration hash"):
            require_preregistration(a_plan(), "0" * 64)

    def test_a_threshold_must_have_been_preregistered(self) -> None:
        plan = a_plan()
        assert plan.threshold_for("forward_return_1h").minimum_absolute_effect == 0.004
        with pytest.raises(B4DataError, match="no practical threshold"):
            plan.threshold_for("forward_return_72h")


class TestPeriodSplit:
    def test_the_holdout_is_the_end_of_the_history_not_a_random_sample(self) -> None:
        split = split_history(BASE, BASE + timedelta(days=1000))
        assert split.development_start == BASE
        assert split.validation_end == BASE + timedelta(days=1000)
        assert split.development_end == split.validation_start
        assert split.validation_is_useful

    def test_a_short_history_yields_an_honest_unusable_holdout(self) -> None:
        """B4.41. Better to say the holdout is too small than to pretend."""
        split = split_history(BASE, BASE + timedelta(days=200))
        assert not split.validation_is_useful
        assert "exploratory" in split.rationale

    def test_rows_are_filtered_to_one_side(self) -> None:
        split = split_history(BASE, BASE + timedelta(days=1000))
        rows = [{"forecast_origin": BASE + timedelta(days=day)} for day in (10, 900)]
        assert len(filter_to_period(rows, split, period="development")) == 1
        assert len(filter_to_period(rows, split, period="validation")) == 1

    def test_an_unknown_period_is_rejected(self) -> None:
        split = split_history(BASE, BASE + timedelta(days=1000))
        with pytest.raises(B4DataError, match="unknown period"):
            filter_to_period([], split, period="test")

    def test_a_backwards_history_is_rejected(self) -> None:
        with pytest.raises(B4DataError, match="after its start"):
            split_history(BASE + timedelta(days=10), BASE)


# --------------------------------------------------------------- the registry


def a_test(**overrides: object) -> CorrectedTest:
    payload: dict[str, object] = {
        "test_id": "sig@1h",
        "family": "f",
        "n": 200,
        "effect": 0.02,
        "lower": 0.01,
        "upper": 0.03,
        "p_value": 0.001,
        "q_value": 0.01,
        "family_size": 8,
        "significant_at_q": True,
    }
    payload.update(overrides)
    return CorrectedTest(**payload)  # type: ignore[arg-type]


POLICY = CarryForwardPolicy()


class TestCarryForwardDecision:
    def test_a_signal_that_clears_every_gate_is_carried_forward(self) -> None:
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=200,
            effective_sample_count=60,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=a_test(),
            evidence=Evidence.SUPPORTED,
            stable=True,
            fragile=False,
            largest_source_share=0.3,
        )
        assert decision is SignalDecision.CARRY_FORWARD
        assert "clears every preregistered gate" in reasons[0]

    def test_no_data_is_insufficient_not_rejected(self) -> None:
        """The distinction the whole registry turns on."""
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=0,
            effective_sample_count=0,
            pit_status=None,
            best=None,
            evidence=None,
            stable=None,
            fragile=None,
            largest_source_share=None,
        )
        assert decision is SignalDecision.INSUFFICIENT_DATA
        assert reasons == ("no observations exist for this signal",)

    def test_an_overlapping_sample_with_too_few_independent_windows_is_insufficient(self) -> None:
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=400,
            effective_sample_count=4,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=a_test(),
            evidence=Evidence.SUPPORTED,
            stable=True,
            fragile=False,
            largest_source_share=0.1,
        )
        assert decision is SignalDecision.INSUFFICIENT_DATA
        assert "effective (non-overlapping)" in reasons[0]

    def test_retrospective_evidence_can_never_be_carried_forward(self) -> None:
        """B4.36: only PIT_VALIDATED data is eligible for later forecasting."""
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=200,
            effective_sample_count=60,
            pit_status=EvidenceTier.RETROSPECTIVE_ONLY,
            best=a_test(),
            evidence=Evidence.SUPPORTED,
            stable=True,
            fragile=False,
            largest_source_share=0.1,
        )
        assert decision is SignalDecision.EXPLORATORY_ONLY
        assert "not PIT_VALIDATED" in reasons[0]

    def test_failing_correction_rejects_and_says_which_clause(self) -> None:
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=200,
            effective_sample_count=60,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=a_test(q_value=0.5),
            evidence=Evidence.WEAK,
            stable=True,
            fragile=False,
            largest_source_share=0.1,
        )
        assert decision is SignalDecision.REJECT
        assert any("q=0.5000 exceeds" in reason for reason in reasons)

    def test_a_fragile_estimate_is_rejected(self) -> None:
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=200,
            effective_sample_count=60,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=a_test(),
            evidence=Evidence.SUPPORTED,
            stable=True,
            fragile=True,
            largest_source_share=0.1,
        )
        assert decision is SignalDecision.REJECT
        assert any("single observation" in reason for reason in reasons)

    def test_one_dominant_source_is_rejected(self) -> None:
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=200,
            effective_sample_count=60,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=a_test(),
            evidence=Evidence.SUPPORTED,
            stable=True,
            fragile=False,
            largest_source_share=0.95,
        )
        assert decision is SignalDecision.REJECT
        assert any("one source" in reason for reason in reasons)

    def test_an_exploratory_carry_requires_an_argued_reason(self) -> None:
        arguments: dict[str, object] = {
            "signal_id": "s",
            "policy": POLICY,
            "sample_count": 200,
            "effective_sample_count": 60,
            "pit_status": EvidenceTier.PIT_VALIDATED,
            "best": a_test(q_value=0.4),
            "evidence": Evidence.WEAK,
            "stable": True,
            "fragile": False,
            "largest_source_share": 0.1,
        }
        assert decide(**arguments)[0] is SignalDecision.REJECT  # type: ignore[arg-type]
        decision, reasons = decide(**arguments, exploratory_reason="large effect, short sample")  # type: ignore[arg-type]
        assert decision is SignalDecision.EXPLORATORY_ONLY
        assert any("carried as exploratory only" in reason for reason in reasons)

    def test_unassessed_stability_blocks_carry_forward(self) -> None:
        decision, reasons = decide(
            signal_id="s",
            policy=POLICY,
            sample_count=200,
            effective_sample_count=60,
            pit_status=EvidenceTier.PIT_VALIDATED,
            best=a_test(),
            evidence=Evidence.SUPPORTED,
            stable=None,
            fragile=False,
            largest_source_share=0.1,
        )
        assert decision is SignalDecision.REJECT
        assert any("could not be assessed" in reason for reason in reasons)


def a_candidate(signal_id: str, decision: SignalDecision, q: float) -> SignalCandidate:
    return SignalCandidate(
        signal_id=signal_id,
        definition="d",
        source_domain="CROSS_ASSET",
        feature_version="v1",
        sample_count=200,
        effective_sample_count=60,
        tested_horizons=("1h",),
        effects={"1h": 0.01},
        intervals={"1h": (0.0, 0.02)},
        q_values={"1h": q},
        evidence={"1h": Evidence.WEAK},
        stability_verdict="STABLE",
        concentration_verdict="DIVERSE",
        largest_source_share=0.2,
        pit_status=EvidenceTier.PIT_VALIDATED,
        multiple_testing_family="f",
        decision=decision,
        decision_reasons=("r",),
    )


class TestRegistry:
    def test_the_exploratory_budget_demotes_the_weakest_visibly(self) -> None:
        """B4.38: do not flood the next track, and do not truncate silently."""
        candidates = [
            a_candidate(f"s{index}", SignalDecision.EXPLORATORY_ONLY, q=0.1 * (index + 1))
            for index in range(6)
        ]
        adjusted = enforce_exploratory_budget(candidates, CarryForwardPolicy(maximum_exploratory_candidates=3))
        kept = [c for c in adjusted if c.decision is SignalDecision.EXPLORATORY_ONLY]
        demoted = [c for c in adjusted if c.decision is SignalDecision.REJECT]
        assert [c.signal_id for c in kept] == ["s0", "s1", "s2"]
        assert len(demoted) == 3
        assert all("budget" in c.decision_reasons[-1] for c in demoted)

    def test_a_budget_that_is_not_exceeded_changes_nothing(self) -> None:
        candidates = [a_candidate("s0", SignalDecision.EXPLORATORY_ONLY, q=0.1)]
        assert enforce_exploratory_budget(candidates, CarryForwardPolicy()) == candidates

    def test_a_registry_round_trips_and_refuses_to_be_overwritten(self, tmp_path: Path) -> None:
        """B4.49: a completed run is never overwritten."""
        registry = SignalRegistry(
            run_id="run-1",
            created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
            preregistration_hash="a" * 64,
            candidates=(a_candidate("s0", SignalDecision.REJECT, q=0.9),),
        )
        path = registry.write(tmp_path / "registry.json")
        assert load_registry(path).run_id == "run-1"
        with pytest.raises(B4DataError, match="never overwritten"):
            registry.write(path)

    def test_the_summary_counts_every_decision_including_the_empty_ones(self) -> None:
        registry = SignalRegistry(
            run_id="run-1",
            created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
            preregistration_hash="a" * 64,
            candidates=(
                a_candidate("s0", SignalDecision.REJECT, q=0.9),
                a_candidate("s1", SignalDecision.INSUFFICIENT_DATA, q=1.0),
            ),
        )
        summary = registry_summary(registry)
        assert summary == {
            "CARRY_FORWARD": 0,
            "EXPLORATORY_ONLY": 0,
            "REJECT": 1,
            "INSUFFICIENT_DATA": 1,
        }
