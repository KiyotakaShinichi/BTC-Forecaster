"""B4.24 / B4.44 — the Granger study behind its gate, and the research plots.

These touch the two optional dependencies (statsmodels, matplotlib) that the
rest of B4 never imports, so they are kept in their own file and skip cleanly
when either is absent. No network.
"""

from __future__ import annotations

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
    granger_study,
    lead_lag_study,
)
from market_intelligence.b4.eventstudy import EventFilter, EventStudySpec, run_event_study
from market_intelligence.b4.stability import decay_profile, period_stability
from market_intelligence.b4.stats import BootstrapConfig

matplotlib = pytest.importorskip("matplotlib", reason="plots are optional research output")
statsmodels = pytest.importorskip("statsmodels", reason="Granger study is optional")

from market_intelligence.b4.plots import (  # noqa: E402 -- must follow the skip guards
    plot_coverage_over_time,
    plot_decay,
    plot_effect_intervals,
    plot_event_response,
    plot_period_stability,
)

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
DAY = 86_400


def series(closes: list[float], series_id: str = "s") -> MarketSeries:
    return MarketSeries(
        series_id=series_id,
        ticker="T",
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


class TestGrangerStudy:
    def test_the_gate_refuses_a_short_sample_and_no_result_is_produced(self) -> None:
        """A refusal is recorded as a finding, not as a missing row."""
        decision, result = granger_study("a", "b", [0.1] * 50, [0.2] * 50, minimum_observations=200)
        assert not decision.run
        assert result is None

    def test_an_injected_lagged_dependence_is_detected(self) -> None:
        """The machinery must be able to find structure that is really there."""
        generator = random.Random(11)
        predictor = [generator.gauss(0.0, 1.0) for _ in range(500)]
        outcome = [0.0]
        for index in range(1, 500):
            outcome.append(0.7 * predictor[index - 1] + generator.gauss(0.0, 0.3))
        decision, result = granger_study("driver", "driven", predictor, outcome)
        assert decision.run
        assert result is not None
        assert result.p_value < 0.01
        assert result.best_lag >= 1

    def test_two_independent_series_show_nothing(self) -> None:
        """And must be able to come back empty."""
        generator = random.Random(12)
        first = [generator.gauss(0.0, 1.0) for _ in range(500)]
        second = [generator.gauss(0.0, 1.0) for _ in range(500)]
        decision, result = granger_study("a", "b", first, second)
        assert decision.run
        assert result is not None
        assert result.p_value > 0.05

    def test_the_result_carries_no_effect_size_and_says_why(self) -> None:
        generator = random.Random(13)
        values = [generator.gauss(0.0, 1.0) for _ in range(400)]
        _, result = granger_study("a", "b", values, values[::-1])
        assert result is not None
        record = result.as_test_record("granger")
        assert record.effect == 0.0 and record.lower == 0.0 and record.upper == 0.0
        assert "not evidence of causation" in result.note
        assert set(result.lag_p_values) == {1, 2, 3, 4, 5}


class TestPlots:
    def _null_study(self) -> object:
        generator = random.Random(3)
        closes = [30_000.0]
        for _ in range(400):
            closes.append(closes[-1] * (1.0 + generator.gauss(0.0, 0.01)))
        from market_intelligence.b4.eventstudy import StudyEvent  # noqa: PLC0415

        events = [
            StudyEvent(
                event_id=f"e{index}",
                event_type="REGULATION",
                entity="SEC",
                event_time=BASE + timedelta(days=10 + index * 5),
                available_at=BASE + timedelta(days=10 + index * 5),
                btc_relevance=0.9,
                confidence=0.9,
                novelty=0.9,
                sentiment=0.0,
                source_ids=(f"d{index}",),
            )
            for index in range(40)
        ]
        spec = EventStudySpec(
            study_id="plot-study",
            event_filter=EventFilter(event_types=("REGULATION",)),
            post_horizons=("1d", "3d"),
            minimum_event_count=10,
            bootstrap=BootstrapConfig(replicates=150, block_length=4, seed=2),
            cluster_window_hours=6,
        )
        return run_event_study(events, spec, series(closes, "btc"))

    def test_an_event_response_figure_is_written(self, tmp_path: Path) -> None:
        path = plot_event_response(self._null_study(), tmp_path / "response.png")  # type: ignore[arg-type]
        assert path.exists() and path.stat().st_size > 0

    def test_an_effect_interval_figure_is_written(self, tmp_path: Path) -> None:
        generator = random.Random(4)
        asset = [100.0]
        btc = [30_000.0]
        for _ in range(400):
            asset.append(asset[-1] * (1.0 + generator.gauss(0.0, 0.02)))
            btc.append(btc[-1] * (1.0 + generator.gauss(0.0, 0.02)))
        spec = LeadLagSpec(
            study_id="ll",
            features=(
                CrossAssetFeatureSpec(
                    feature_id="a1", series_id="a", kind=CrossAssetFeatureKind.LAGGED_RETURN, window="1d"
                ),
            ),
            horizons=("1d",),
            minimum_observations=50,
            bootstrap=BootstrapConfig(replicates=120, block_length=4, seed=1),
        )
        origins = [BASE + timedelta(days=10 + index) for index in range(300)]
        results = lead_lag_study(spec, series(btc, "btc"), {"a": series(asset, "a")}, origins)
        path = plot_effect_intervals(results, tmp_path / "effects.png")
        assert path.exists() and path.stat().st_size > 0

    def test_plotting_nothing_is_an_error_not_an_empty_chart(self, tmp_path: Path) -> None:
        """An empty axis reads as 'no effect'; an error reads as 'no data'."""
        with pytest.raises(B4DataError, match="no estimable cells"):
            plot_effect_intervals([], tmp_path / "empty.png")

    def test_a_decay_figure_is_written(self, tmp_path: Path) -> None:
        profile = decay_profile("s", {"1h": [0.02] * 5, "6h": [0.01] * 5}, ["1h", "6h"])
        assert plot_decay(profile, tmp_path / "decay.png").exists()

    def test_an_empty_decay_profile_is_refused(self, tmp_path: Path) -> None:
        profile = decay_profile("s", {}, ["1h", "6h"])
        with pytest.raises(B4DataError, match="no horizon"):
            plot_decay(profile, tmp_path / "decay.png")

    def test_a_period_stability_figure_is_written(self, tmp_path: Path) -> None:
        observations = [
            (datetime(year, 6, 1, tzinfo=timezone.utc) + timedelta(hours=index), 0.01)
            for year in (2022, 2023)
            for index in range(30)
        ]
        result = period_stability(observations)
        assert plot_period_stability(result, tmp_path / "stability.png", title="test").exists()

    def test_a_coverage_figure_needs_matching_lengths(self, tmp_path: Path) -> None:
        origins = [BASE + timedelta(days=index) for index in range(10)]
        assert plot_coverage_over_time(origins, [1.0] * 10, tmp_path / "coverage.png").exists()
        with pytest.raises(B4DataError, match="same length"):
            plot_coverage_over_time(origins, [1.0] * 3, tmp_path / "bad.png")
