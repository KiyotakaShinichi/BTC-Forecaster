"""Point-in-time regime labelling and per-regime performance.

The central test is the causality one: a regime label computed from the whole
series and back-applied to the past is look-ahead, and conditioning results on
it would silently invalidate every per-regime claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.backtesting.engine import run_walk_forward
from btc_forecaster.backtesting.splits import WalkForwardSplitter
from btc_forecaster.evaluation.regimes import (
    RegimeConfig,
    TrendRegime,
    VolatilityRegime,
    assert_regime_labels_are_causal,
    label_fold_origins,
    label_regimes,
    performance_by_regime,
    realised_volatility,
    regime_at,
    trailing_return,
)
from btc_forecaster.models.baselines import RandomWalk, RandomWalkWithDrift
from btc_forecaster.testing import constant_growth_frame, synthetic_market_frame
from btc_forecaster.timebase import UTC


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_market_frame(periods=1200, seed=17)


class TestRegimeLabelsAreCausal:
    """A whole-series label back-applied to the past is look-ahead."""

    def test_labels_ignore_the_future(self, frame):
        assert_regime_labels_are_causal(frame, cut=-120)

    def test_labels_on_a_prefix_match_labels_on_the_whole_series(self, frame):
        cut = 900
        full = label_regimes(frame).iloc[:cut]
        prefix = label_regimes(frame.iloc[:cut])
        pd.testing.assert_frame_equal(full, prefix, check_freq=False)

    def test_the_causality_check_detects_a_whole_series_quantile(self, frame):
        """The exact mistake the module exists to prevent, proved detectable."""
        config = RegimeConfig()

        def leaky_labels(f: pd.DataFrame, cfg=None) -> pd.DataFrame:
            vol = realised_volatility(f, config.vol_window)
            threshold = vol.quantile(config.vol_quantile)  # whole-series!
            out = pd.DataFrame(index=f.index)
            out["realised_vol"] = vol
            out["vol_threshold"] = threshold
            out["trailing_return"] = trailing_return(f, config.trend_window)
            regime = pd.Series(pd.NA, index=f.index, dtype="object")
            regime[vol.notna() & (vol > threshold)] = VolatilityRegime.HIGH.value
            regime[vol.notna() & (vol <= threshold)] = VolatilityRegime.LOW.value
            out["volatility_regime"] = regime
            out["trend_regime"] = regime
            out["regime"] = regime
            return out

        import btc_forecaster.evaluation.regimes as regimes_module

        original = regimes_module.label_regimes
        regimes_module.label_regimes = leaky_labels
        try:
            with pytest.raises(AssertionError, match="not causal"):
                assert_regime_labels_are_causal(frame, cut=-120)
        finally:
            regimes_module.label_regimes = original

    def test_regime_at_uses_only_bars_up_to_the_origin(self, frame):
        origin = frame.index[800]
        from_full = regime_at(frame, origin)
        from_truncated = regime_at(frame.iloc[:801], origin)
        assert from_full == from_truncated

    def test_origin_before_all_data_is_an_error(self, frame):
        with pytest.raises(ValueError, match="no observations"):
            regime_at(frame, pd.Timestamp("2000-01-01", tz=UTC))


class TestLabelSemantics:
    def test_early_bars_are_unlabelled_rather_than_guessed(self, frame):
        labels = label_regimes(frame)
        assert labels["volatility_regime"].iloc[:300].isna().all()

    def test_every_late_bar_gets_a_label(self, frame):
        labels = label_regimes(frame)
        assert labels["regime"].iloc[600:].notna().all()

    def test_a_trending_series_is_labelled_bull(self):
        frame = constant_growth_frame(periods=800, daily_growth=0.003)
        labels = label_regimes(frame)
        assert labels["trend_regime"].iloc[-1] == TrendRegime.BULL.value

    def test_a_declining_series_is_labelled_bear(self):
        frame = constant_growth_frame(periods=800, daily_growth=-0.003)
        labels = label_regimes(frame)
        assert labels["trend_regime"].iloc[-1] == TrendRegime.BEAR.value

    def test_a_flat_series_is_labelled_sideways(self):
        frame = constant_growth_frame(periods=800, daily_growth=0.0)
        labels = label_regimes(frame)
        assert labels["trend_regime"].iloc[-1] == TrendRegime.SIDEWAYS.value

    def test_volatility_is_split_near_the_configured_quantile(self, frame):
        labels = label_regimes(frame, RegimeConfig(vol_quantile=0.5)).dropna(subset=["regime"])
        share_high = (labels["volatility_regime"] == VolatilityRegime.HIGH.value).mean()
        assert 0.3 < share_high < 0.7, "a median split should be roughly balanced"

    def test_the_volatility_threshold_is_relative_not_absolute(self):
        """BTC's 'high volatility' in 2017 and 2025 are different absolute
        numbers; a fixed cutoff would mostly label the calendar."""
        calm = synthetic_market_frame(periods=800, daily_vol=0.005, seed=1)
        wild = synthetic_market_frame(periods=800, daily_vol=0.08, seed=1)

        calm_labels = label_regimes(calm).dropna(subset=["regime"])
        wild_labels = label_regimes(wild).dropna(subset=["regime"])

        calm_high = (calm_labels["volatility_regime"] == VolatilityRegime.HIGH.value).mean()
        wild_high = (wild_labels["volatility_regime"] == VolatilityRegime.HIGH.value).mean()
        assert abs(calm_high - wild_high) < 0.2, "the split should not depend on absolute scale"

    def test_combined_regime_joins_both_axes(self, frame):
        labels = label_regimes(frame).dropna(subset=["regime"])
        sample = labels["regime"].iloc[-1]
        volatility, trend = sample.split("/")
        assert volatility in {r.value for r in VolatilityRegime}
        assert trend in {r.value for r in TrendRegime}

    def test_config_is_serialisable_and_self_describing(self):
        import json

        payload = RegimeConfig().to_dict()
        json.dumps(payload)
        assert "trailing data only" in payload["definition"]


class TestFoldOriginLabelling:
    def test_one_row_per_origin(self, frame):
        splitter = WalkForwardSplitter(horizon=20, n_folds=6, min_train_bars=500)
        origins = [fold.origin.last_observed_bar for fold in splitter.split(frame.index)]
        labelled = label_fold_origins(frame, origins)

        assert len(labelled) == 6
        assert list(labelled["origin"]) == origins

    def test_each_origin_label_matches_a_standalone_computation(self, frame):
        origins = [frame.index[700], frame.index[900]]
        labelled = label_fold_origins(frame, origins)
        for _, row in labelled.iterrows():
            assert row["regime"] == regime_at(frame, row["origin"])["regime"]


class TestPerformanceByRegime:
    @pytest.fixture
    def scored(self, frame):
        splitter = WalkForwardSplitter(horizon=15, n_folds=12, min_train_bars=500)
        result = run_walk_forward(frame, [RandomWalk(), RandomWalkWithDrift()], splitter)
        origins = [fold.origin.last_observed_bar for fold in result.folds]
        return result.prediction_records(), label_fold_origins(frame, origins)

    def test_it_splits_results_by_regime_and_model(self, scored):
        records, regimes = scored
        table = performance_by_regime(records, regimes)

        assert table.index.names == ["model", "regime"]
        assert {"mae", "rmse", "directional_accuracy", "n", "n_origins"} <= set(table.columns)
        assert set(table.index.get_level_values("model")) == {"random_walk", "random_walk_drift"}

    def test_origin_counts_sum_to_the_number_of_folds(self, scored):
        records, regimes = scored
        table = performance_by_regime(records, regimes)
        for model in table.index.get_level_values("model").unique():
            total = table.loc[model, "n_origins"].sum()
            assert total == regimes["regime"].notna().sum()

    def test_it_can_split_on_a_single_axis(self, scored):
        records, regimes = scored
        table = performance_by_regime(records, regimes, by="volatility_regime")
        assert set(table.index.get_level_values("volatility_regime")) <= {
            r.value for r in VolatilityRegime
        }

    def test_an_unknown_axis_is_reported(self, scored):
        records, regimes = scored
        with pytest.raises(ValueError, match="not a regime column"):
            performance_by_regime(records, regimes, by="nonexistent")

    def test_unlabelled_origins_are_excluded_not_bucketed(self, scored):
        records, regimes = scored
        blanked = regimes.copy()
        blanked.loc[blanked.index[0], "regime"] = None
        table = performance_by_regime(records, blanked)
        for model in table.index.get_level_values("model").unique():
            assert table.loc[model, "n_origins"].sum() == len(blanked) - 1

    def test_metrics_are_finite(self, scored):
        records, regimes = scored
        table = performance_by_regime(records, regimes)
        assert np.isfinite(table["mae"]).all()
        assert table["directional_accuracy"].between(0.0, 1.0).all()
