"""Nested model selection and the XGBOOST_CAUSAL_RETUNED challenger.

The tests that matter are the ones proving selection never reaches the outer
test window, and that the inner splits are themselves forward and embargoed --
a random k-fold inside the training history would train on 2025 to validate on
2021 and is just as invalid as leaking the outer fold.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from btc_forecaster.backtesting.engine import run_walk_forward
from btc_forecaster.backtesting.splits import WalkForwardSplitter
from btc_forecaster.models.base import TrainingWindow
from btc_forecaster.models.tuning import (
    SelectionRecord,
    inner_validation_splits,
    sample_parameter_grid,
    select_by_inner_validation,
)
from btc_forecaster.testing import synthetic_market_frame

has_xgboost = importlib.util.find_spec("xgboost") is not None
needs_xgboost = pytest.mark.skipif(not has_xgboost, reason="xgboost not installed")


class TestInnerSplitIntegrity:
    def test_validation_always_follows_training(self):
        """A random k-fold would train on the future to validate on the past."""
        for split in inner_validation_splits(1000, n_splits=3, val_bars=90, embargo=5):
            assert split.train_end < split.val_start

    def test_the_embargo_gap_is_applied(self):
        for split in inner_validation_splits(1000, n_splits=3, val_bars=90, embargo=7):
            assert split.val_start - split.train_end - 1 == 7

    def test_zero_embargo_puts_validation_immediately_after_training(self):
        for split in inner_validation_splits(1000, n_splits=2, val_bars=60, embargo=0):
            assert split.val_start == split.train_end + 1

    def test_no_split_reaches_past_the_training_history(self):
        n = 1000
        for split in inner_validation_splits(n, n_splits=4, val_bars=60, embargo=5):
            assert split.val_end <= n - 1
            assert split.train_start >= 0

    def test_splits_are_expanding_and_ordered(self):
        splits = inner_validation_splits(1200, n_splits=3, val_bars=90, embargo=5)
        assert [s.train_end for s in splits] == sorted(s.train_end for s in splits)
        assert splits[-1].n_train > splits[0].n_train

    def test_validation_windows_do_not_overlap(self):
        splits = inner_validation_splits(1200, n_splits=3, val_bars=90, embargo=5)
        spans = [(s.val_start, s.val_end) for s in splits]
        # Pairwise over consecutive splits: strict=False, the lists differ in
        # length by one by construction.
        for (_, earlier_end), (later_start, _) in zip(spans, spans[1:], strict=False):
            assert earlier_end < later_start
        assert len({s.val_start for s in splits}) == len(splits)

    def test_the_most_recent_data_is_always_used_for_validation(self):
        n = 1000
        splits = inner_validation_splits(n, n_splits=3, val_bars=90, embargo=5)
        assert max(s.val_end for s in splits) == n - 1

    def test_insufficient_history_is_a_clear_error(self):
        with pytest.raises(ValueError, match="need at least"):
            inner_validation_splits(100, n_splits=3, val_bars=90, embargo=5, min_train_bars=250)

    def test_split_counts_are_reported(self):
        split = inner_validation_splits(1000, n_splits=1, val_bars=60, embargo=3)[0]
        assert split.n_val == 60
        assert split.to_dict()["embargo"] == 3

    @pytest.mark.parametrize("bad", [{"n_splits": 0}, {"val_bars": 0}])
    def test_invalid_configuration_is_rejected(self, bad):
        with pytest.raises(ValueError):
            inner_validation_splits(1000, **bad)


class TestParameterSampling:
    def test_sampling_is_deterministic_under_a_seed(self):
        space = {"a": [1, 2, 3], "b": [0.1, 0.2]}
        assert sample_parameter_grid(space, n_candidates=4, seed=7) == sample_parameter_grid(
            space, n_candidates=4, seed=7
        )

    def test_candidates_are_distinct(self):
        space = {"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]}
        candidates = sample_parameter_grid(space, n_candidates=6, seed=0)
        keys = {tuple(sorted(c.items())) for c in candidates}
        assert len(keys) == len(candidates)

    def test_it_cannot_exceed_the_size_of_the_space(self):
        space = {"a": [1, 2]}
        assert len(sample_parameter_grid(space, n_candidates=99, seed=0)) == 2

    def test_every_candidate_covers_every_dimension(self):
        space = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        for candidate in sample_parameter_grid(space, n_candidates=4, seed=1):
            assert set(candidate) == {"a", "b", "c"}

    def test_zero_candidates_is_rejected(self):
        with pytest.raises(ValueError, match="n_candidates"):
            sample_parameter_grid({"a": [1]}, n_candidates=0)


class TestSelectByInnerValidation:
    def test_it_picks_the_lowest_mean_loss(self):
        candidates = [{"k": 1}, {"k": 2}, {"k": 3}]
        record = select_by_inner_validation(
            1000, candidates, lambda p, s: float(p["k"]), n_splits=3, val_bars=90, embargo=5
        )
        assert record.chosen == {"k": 1}
        assert record.chosen_score == pytest.approx(1.0)

    def test_the_ranking_is_ordered_best_first(self):
        candidates = [{"k": 3}, {"k": 1}, {"k": 2}]
        record = select_by_inner_validation(
            1000, candidates, lambda p, s: float(p["k"]), n_splits=2, val_bars=90, embargo=0
        )
        scores = [r["mean_score"] for r in record.ranking]
        assert scores == sorted(scores)

    def test_fit_score_only_ever_sees_indices_inside_the_training_history(self):
        """The nested-selection guarantee, asserted on the indices handed out."""
        n_rows = 1000
        seen = []

        def spy(params, split):
            seen.append((split.train_start, split.train_end, split.val_start, split.val_end))
            return 1.0

        select_by_inner_validation(
            n_rows, [{"a": 1}], spy, n_splits=3, val_bars=90, embargo=5
        )

        assert seen
        for train_start, train_end, val_start, val_end in seen:
            assert train_start >= 0
            assert val_end <= n_rows - 1
            assert train_end < val_start

    def test_a_failing_candidate_is_dropped_with_a_note(self):
        def flaky(params, split):
            if params["k"] == 2:
                raise RuntimeError("did not converge")
            return float(params["k"])

        record = select_by_inner_validation(
            1000, [{"k": 1}, {"k": 2}], flaky, n_splits=2, val_bars=90, embargo=0
        )
        assert record.chosen == {"k": 1}
        assert any("did not converge" in note for note in record.notes)

    def test_all_candidates_failing_is_an_error(self):
        def broken(params, split):
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="no candidate configuration"):
            select_by_inner_validation(1000, [{"k": 1}], broken, n_splits=2, val_bars=90)

    def test_an_indiscriminate_search_says_so(self):
        """A tuned model whose configurations are indistinguishable has not
        earned the compute it took to find one."""
        candidates = [{"k": 1.0000}, {"k": 1.0001}]
        record = select_by_inner_validation(
            1000, candidates, lambda p, s: float(p["k"]), n_splits=2, val_bars=90
        )
        assert any("did not meaningfully discriminate" in note for note in record.notes)

    def test_score_spread_quantifies_how_much_the_search_mattered(self):
        record = select_by_inner_validation(
            1000, [{"k": 1}, {"k": 5}], lambda p, s: float(p["k"]), n_splits=2, val_bars=90
        )
        assert record.score_spread == pytest.approx(4.0)

    def test_the_record_is_serialisable(self):
        record = select_by_inner_validation(
            1000, [{"k": 1}], lambda p, s: 1.0, n_splits=2, val_bars=90
        )
        json.dumps(record.to_dict())

    def test_no_candidates_is_an_error(self):
        with pytest.raises(ValueError, match="no candidate"):
            select_by_inner_validation(1000, [], lambda p, s: 1.0)

    def test_the_record_names_its_scoring_rule(self):
        record = select_by_inner_validation(
            1000, [{"k": 1}], lambda p, s: 1.0, n_splits=1, val_bars=90, scoring="custom loss"
        )
        assert record.scoring == "custom loss"
        assert isinstance(record, SelectionRecord)


@needs_xgboost
@pytest.mark.slow
class TestXgboostCausalRetuned:
    @pytest.fixture(scope="class")
    def window(self) -> TrainingWindow:
        return TrainingWindow(synthetic_market_frame(periods=900, seed=3))

    @pytest.fixture(scope="class")
    def fitted(self, window):
        from btc_forecaster.models import registry

        model = registry.build("xgboost_causal_retuned", monte_carlo_runs=200)
        model.fit(window)
        return model

    def test_it_is_named_after_the_registry_key(self, fitted):
        assert fitted.name == "xgboost_causal_retuned"

    def test_it_satisfies_the_forecast_contract(self, fitted, window):
        result = fitted.predict(20)
        assert result.index.equals(window.origin.target_bars(20))
        assert (result.point > 0).all()
        assert (result.lower <= result.point).all()
        assert (result.point <= result.upper).all()

    def test_intervals_widen_with_horizon(self, fitted):
        result = fitted.predict(60)
        width = np.log(result.upper) - np.log(result.lower)
        assert width.iloc[-1] > width.iloc[0] * 2

    def test_it_targets_returns_not_price_residuals(self, fitted):
        assert fitted.predict(5).metadata["target"] == "next-bar log return"

    def test_tuning_happens_inside_fit(self, fitted, window):
        record = fitted.tuning
        assert record.n_candidates == 12
        assert record.n_splits == 3
        assert record.train_rows < len(window), "selection used the supervised rows only"

    def test_boost_rounds_are_fitted_by_early_stopping_not_searched(self, fitted):
        from btc_forecaster.models.challengers import DEFAULT_SEARCH_SPACE

        assert "num_boost_round" not in DEFAULT_SEARCH_SPACE
        assert "num_boost_round" not in fitted.tuning.chosen
        assert fitted._chosen_rounds >= 1

    def test_on_a_random_walk_the_search_barely_discriminates(self, fitted):
        """The honest outcome on data with no signal, and it must be visible."""
        assert fitted.tuning.score_spread < 0.01

    def test_selected_features_are_persisted(self, fitted):
        assert fitted.tuning.features
        assert fitted.describe()["features"]["fitted_on"]["n_observations"] > 0

    def test_describe_is_serialisable_and_records_the_search(self, fitted):
        described = fitted.describe()
        json.dumps(described)
        assert described["tuning"]["n_candidates"] == 12
        assert "search_space" in described

    def test_refitting_reselects_on_the_new_window(self, window):
        from btc_forecaster.models import registry

        model = registry.build("xgboost_causal_retuned", monte_carlo_runs=100, n_candidates=4)
        model.fit(window)
        first = model._selection.fitted_end

        model.fit(TrainingWindow(window.frame.iloc[:700]))
        assert model._selection.fitted_end < first

    def test_too_little_data_is_refused(self, window):
        from btc_forecaster.models import registry

        model = registry.build("xgboost_causal_retuned")
        with pytest.raises(ValueError, match="at least"):
            model.fit(TrainingWindow(window.frame.iloc[:200]))


@needs_xgboost
@pytest.mark.slow
class TestNestedSelectionInsideTheBacktest:
    def test_the_model_never_receives_the_outer_test_window(self):
        """A2.3 end to end: what fit() actually got, per outer fold."""
        from btc_forecaster.models.challengers import XgboostCausalRetuned

        seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        class Spy(XgboostCausalRetuned):
            def _fit(self, window: TrainingWindow) -> None:
                seen.append((window.index.min(), window.index.max()))
                super()._fit(window)

        frame = synthetic_market_frame(periods=1100, seed=9)
        splitter = WalkForwardSplitter(horizon=20, n_folds=2, min_train_bars=600, embargo_bars=5)
        folds = splitter.split(frame.index)

        run_walk_forward(
            frame,
            [Spy(name="spy", monte_carlo_runs=100, n_candidates=3)],
            splitter,
            baseline="spy",
        )

        assert len(seen) == len(folds)
        for (start, end), fold in zip(seen, folds, strict=True):
            assert start == fold.train_start
            assert end == fold.train_end
            assert end < fold.test_start, "selection must not reach the outer test window"

    def test_each_fold_records_its_own_selection(self):
        from btc_forecaster.models.challengers import XgboostCausalRetuned

        records: list[dict] = []

        class Recording(XgboostCausalRetuned):
            def _fit(self, window: TrainingWindow) -> None:
                super()._fit(window)
                records.append(self.tuning.to_dict())

        frame = synthetic_market_frame(periods=1100, seed=10)
        splitter = WalkForwardSplitter(horizon=20, n_folds=2, min_train_bars=600)
        run_walk_forward(
            frame,
            [Recording(name="rec", monte_carlo_runs=100, n_candidates=3)],
            splitter,
            baseline="rec",
        )

        assert len(records) == 2
        assert records[0]["train_rows"] < records[1]["train_rows"], "expanding window"
        for record in records:
            json.dumps(record)
            assert record["features"]
