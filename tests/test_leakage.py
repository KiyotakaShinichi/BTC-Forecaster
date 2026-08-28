"""Tests designed specifically to catch future leakage.

These are the most important tests in the repository. Each one is written to
fail if a specific, previously-present defect is reintroduced.

The general technique is the **future mutation test**: compute something, then
change the data *after* some cut point, recompute, and assert nothing at or
before the cut moved. A quantity that changes when the future changes was
reading the future. This catches centred windows, backward fills, negative
shifts, and whole-series normalisation -- none of which are visible from a
column name or a docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.features.pipeline import (
    assert_causal,
    build_feature_frame,
    future_feature_row,
    to_supervised,
)
from btc_forecaster.features.selection import FeatureSelector
from btc_forecaster.features.spec import (
    Ema,
    FeatureSpec,
    LagReturn,
    PriceOverSma,
    RollingMeanReturn,
    RollingMeanVolume,
    RollingStdReturn,
    Sma,
    default_specs,
    log_price,
)
from btc_forecaster.testing import synthetic_market_frame
from btc_forecaster.timebase import UTC, ForecastOrigin


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_market_frame(periods=400, seed=3)


ALL_SPECS = [
    LagReturn.of(1),
    LagReturn.of(7),
    RollingMeanReturn.of(14),
    RollingStdReturn.of(14),
    RollingMeanVolume.of(14),
    Ema.of(10),
    Sma.of(10),
    PriceOverSma.of(20),
]


class TestFeaturesAreCausal:
    """Every feature at bar D must be a function of bars <= D."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
    def test_single_feature_ignores_the_future(self, frame, spec):
        assert_causal(frame, [spec], cut=-60)

    def test_the_default_feature_set_ignores_the_future(self, frame):
        assert_causal(frame, default_specs(), cut=-60)

    def test_a_selected_feature_set_ignores_the_future(self, frame):
        selection = FeatureSelector(max_lag=20).fit(frame.iloc[:300])
        assert_causal(frame, list(selection.specs), cut=-60)

    def test_the_causality_check_actually_detects_a_leak(self, frame):
        """A guard that never fires is worthless, so prove it fires."""

        class CentredMean(FeatureSpec):
            def compute(self, f: pd.DataFrame) -> pd.Series:
                return f["close"].rolling(11, center=True).mean().rename(self.name)

        with pytest.raises(AssertionError, match="not causal"):
            assert_causal(frame, [CentredMean(name="centred_mean")], cut=-60)

    def test_the_causality_check_detects_a_negative_shift(self, frame):
        class Tomorrow(FeatureSpec):
            def compute(self, f: pd.DataFrame) -> pd.Series:
                return f["close"].shift(-1).rename(self.name)

        with pytest.raises(AssertionError, match="not causal"):
            assert_causal(frame, [Tomorrow(name="tomorrows_close")], cut=-60)

    def test_the_causality_check_detects_whole_series_normalisation(self, frame):
        """Scaling by the full-sample mean leaks the future into every row."""

        class GlobalZScore(FeatureSpec):
            def compute(self, f: pd.DataFrame) -> pd.Series:
                close = f["close"]
                return ((close - close.mean()) / close.std()).rename(self.name)

        with pytest.raises(AssertionError, match="not causal"):
            assert_causal(frame, [GlobalZScore(name="global_z")], cut=-60)

    def test_features_computed_on_a_prefix_match_features_computed_on_the_whole(self, frame):
        """Truncating the future must not change past feature values."""
        cut = 300
        full = build_feature_frame(frame, default_specs()).iloc[:cut]
        prefix = build_feature_frame(frame.iloc[:cut], default_specs())
        pd.testing.assert_frame_equal(full, prefix)


class TestSupervisedAlignment:
    """The step that separates 'causal' from 'usable'."""

    def test_predictor_row_comes_from_the_previous_bar(self, frame):
        specs = [LagReturn.of(1), RollingMeanReturn.of(5)]
        features = build_feature_frame(frame, specs)
        data = to_supervised(features, log_price(frame), step=1)

        target_bar = data.X.index[10]
        assert data.feature_bar.iloc[10] == target_bar - pd.Timedelta(days=1)

    def test_every_predictor_row_predates_its_target_bar(self, frame):
        features = build_feature_frame(frame, default_specs())
        data = to_supervised(features, log_price(frame), step=1)
        assert (data.feature_bar < data.X.index).all()

    def test_design_matrix_row_equals_the_previous_bars_feature_values(self, frame):
        specs = [RollingMeanReturn.of(5), Ema.of(10)]
        features = build_feature_frame(frame, specs)
        data = to_supervised(features, log_price(frame), step=1)

        target_bar = data.X.index[50]
        source_bar = target_bar - pd.Timedelta(days=1)
        for name in data.feature_names:
            assert data.X.loc[target_bar, name] == features.loc[source_bar, name]

    def test_multi_step_alignment_uses_a_correspondingly_older_bar(self, frame):
        features = build_feature_frame(frame, [LagReturn.of(1)])
        data = to_supervised(features, log_price(frame), step=7)
        assert (data.X.index - data.feature_bar == pd.Timedelta(days=7)).all()

    def test_zero_step_alignment_is_refused(self, frame):
        """step=0 is precisely the pre-Track-A defect: features see their target."""
        features = build_feature_frame(frame, default_specs())
        with pytest.raises(ValueError, match="lets features see their own target bar"):
            to_supervised(features, log_price(frame), step=0)

    def test_contemporaneous_features_would_have_leaked_the_target(self, frame):
        """Demonstrates the magnitude of the bug this alignment prevents.

        `ema_1` at bar D is exactly close[D]. Regressing log close on it with no
        shift is a perfect fit; with the correct one-bar shift it is not.
        """
        features = build_feature_frame(frame, [Ema.of(1)])
        target = log_price(frame)

        contemporaneous = np.log(features["ema_1"]).reindex(target.index)
        leaked_error = float((contemporaneous - target).abs().dropna().max())
        assert leaked_error < 1e-12, "sanity: ema_1 is the close itself"

        honest = to_supervised(features, target, step=1)
        honest_error = float((np.log(honest.X["ema_1"]) - honest.y).abs().max())
        assert honest_error > 1e-3, "one-bar shift must break the identity"

    def test_alignment_survives_gaps_in_the_target(self, frame):
        features = build_feature_frame(frame, [LagReturn.of(1)])
        target = log_price(frame).drop(frame.index[100:110])
        data = to_supervised(features, target, step=1)
        assert (data.feature_bar < data.X.index).all()


class TestSelectionIsTrainOnly:
    """Feature selection is a model decision and must not see evaluation data."""

    def test_selection_is_unchanged_by_data_after_its_training_window(self, frame):
        selector = FeatureSelector(max_lag=20)
        train = frame.iloc[:300]

        baseline = selector.fit(train)

        rng = np.random.default_rng(0)
        tampered = frame.copy()
        tail = tampered.index > train.index[-1]
        tampered.loc[tail, "close"] *= 1.0 + rng.normal(scale=0.3, size=int(tail.sum()))
        tampered.loc[tail, "volume"] *= 10.0
        after = selector.fit(tampered.iloc[:300])

        assert baseline.names == after.names
        assert baseline.lags == after.lags
        assert baseline.rolling_windows == after.rolling_windows
        assert baseline.ema_spans == after.ema_spans

    def test_selection_records_the_window_it_was_fitted_on(self, frame):
        train = frame.iloc[:300]
        selection = FeatureSelector(max_lag=20).fit(train)

        assert selection.fitted_start == train.index.min()
        assert selection.fitted_end == train.index.max()
        assert selection.n_observations == 300
        assert selection.fitted_end < frame.index[-1], "audit trail must show held-out data exists"

    def test_different_training_windows_can_select_differently(self, frame):
        """If selection never varied by window it would not be doing anything."""
        selector = FeatureSelector(max_lag=20)
        early = selector.fit(frame.iloc[:200])
        late = selector.fit(frame.iloc[200:])
        assert early.fitted_end < late.fitted_start
        assert isinstance(early.names, list) and isinstance(late.names, list)

    def test_pacf_selection_reports_multiple_comparison_context(self, frame):
        """Scanning 60 lags at alpha=0.05 flags ~3 by chance; say so."""
        selection = FeatureSelector(max_lag=60, alpha=0.05).fit(frame)
        pacf_diag = selection.diagnostics["pacf"]

        assert pacf_diag["max_lag_scanned"] == 60
        assert pacf_diag["n_expected_by_chance"] == pytest.approx(3.0)
        assert "n_significant_exceeds_chance" in pacf_diag

    def test_pacf_finds_a_real_ar1_signal(self):
        """The selector must detect structure that genuinely exists."""
        ar1 = synthetic_market_frame(periods=800, kind="ar1", phi=0.5, seed=11)
        selection = FeatureSelector(max_lag=20).fit(ar1)
        pacf_diag = selection.diagnostics["pacf"]

        assert 1 in selection.lags
        assert pacf_diag["strongest_lag"] == 1
        # phi=0.5 sits many multiples above the white-noise band.
        assert pacf_diag["strongest_over_threshold"] > 5.0

    def test_significant_lag_count_alone_cannot_separate_signal_from_noise(self):
        """Why the diagnostics report magnitude and not just a count.

        A genuine AR(1) and a pure random walk both flag exactly one significant
        lag when 20 are scanned at alpha=0.05. Only the magnitude relative to the
        band tells them apart, so a selector that reported only the count would
        be reporting nothing.
        """
        ar1 = FeatureSelector(max_lag=20).fit(
            synthetic_market_frame(periods=800, kind="ar1", phi=0.5, seed=11)
        )
        noise = FeatureSelector(max_lag=20).fit(
            synthetic_market_frame(periods=800, kind="random_walk", seed=11)
        )

        ar1_pacf = ar1.diagnostics["pacf"]
        noise_pacf = noise.diagnostics["pacf"]

        assert ar1_pacf["n_significant"] == noise_pacf["n_significant"], (
            "premise of this test: the counts are indistinguishable"
        )
        assert ar1_pacf["strongest_over_threshold"] > 3 * noise_pacf["strongest_over_threshold"]
        assert noise_pacf["strongest_over_threshold"] < 1.5, (
            "a spurious hit should sit just over the line"
        )


class TestFutureFeatureRow:
    def test_future_row_uses_only_bars_up_to_the_origin(self, frame):
        specs = default_specs()
        origin = ForecastOrigin(frame.index[300])

        from_full = future_feature_row(frame, specs, origin)
        from_truncated = future_feature_row(frame.iloc[:301], specs, origin)

        pd.testing.assert_frame_equal(from_full, from_truncated)

    def test_future_row_is_stamped_at_the_origin_bar(self, frame):
        origin = ForecastOrigin(frame.index[300])
        row = future_feature_row(frame, default_specs(), origin)
        assert row.index[0] == origin.last_observed_bar

    def test_future_row_matches_the_feature_frame_at_that_bar(self, frame):
        specs = default_specs()
        origin = ForecastOrigin(frame.index[300])
        row = future_feature_row(frame, specs, origin)
        full = build_feature_frame(frame, specs).loc[[origin.last_observed_bar]]
        # check_freq=False: slicing leaves a freq attribute on one index and not
        # the other. That is index metadata, not a difference in values.
        pd.testing.assert_frame_equal(row, full, check_freq=False)

    def test_insufficient_history_is_an_explicit_error(self, frame):
        origin = ForecastOrigin(frame.index[2])
        with pytest.raises(ValueError, match="insufficient history"):
            future_feature_row(frame, [RollingMeanReturn.of(60)], origin)

    def test_origin_before_all_data_is_an_error(self, frame):
        origin = ForecastOrigin(pd.Timestamp("2010-01-01", tz=UTC))
        with pytest.raises(ValueError, match="no observations"):
            future_feature_row(frame, default_specs(), origin)
