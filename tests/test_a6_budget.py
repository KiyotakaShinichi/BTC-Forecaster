"""The training budget binds every model, not only the ones that read ``X``.

The 1,000-row constraint is the premise of the whole track, and it was not
enforced. ``TrainingSet.series`` carried every bar from the start of the data up
to the budget's end, on the assumption that series models would slice it
themselves. The state-space adapters clipped only the upper end, so ``arima``,
``local_level``, ``holt_linear_trend``, ``local_linear_trend`` and
``uc_stochastic_cycle`` estimated on ~2,300 rows at a budget of 250, 500 or
1,000 -- their forecasts were bit-identical across all three -- while the
training fingerprint, computed from ``X`` and ``y``, recorded the budget.

Found because the sample-efficiency study showed ARIMA's skill unchanged to five
decimals at every budget. Nothing else would have shown it: not leakage (the
upper bound held), not a failure, not a warning. These tests make the bound a
property that is checked rather than one that is described.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.adapters.statistical import StateSpaceModel
from btc_forecaster.research.contracts import Family, ModelStatus, TrainingSet
from btc_forecaster.research.partition import PartitionSpec, build_dataset
from btc_forecaster.testing import synthetic_market_frame

FRAME = synthetic_market_frame(periods=500, kind="ar1", seed=41)
SMALL, LARGE = 120, 240


def runnable(*, exclude_family: Family | None = None) -> list[str]:
    """Every model this environment can run, from the registry -- no second list."""
    return [
        m
        for m in registry.model_ids()
        if registry.get(m).effective_status() is ModelStatus.ACTIVE
        and registry.get(m).family is not exclude_family
    ]


def state_space_models() -> list[str]:
    return [m for m in runnable() if isinstance(registry.build(m), StateSpaceModel)]


@pytest.fixture(scope="module")
def datasets():
    return {n: build_dataset(FRAME, spec=PartitionSpec(train_rows=n)) for n in (SMALL, LARGE)}


class TestTheTrainingSetHoldsOnlyTheBudget:
    @pytest.mark.parametrize("n", [SMALL, LARGE])
    def test_series_and_close_cover_exactly_the_budget_bars(self, datasets, n) -> None:
        train = datasets[n].training_set()
        assert len(train.y) == n
        assert train.series.index.equals(train.y.index)
        assert train.close.index.equals(train.y.index)

    def test_the_series_is_the_target_itself(self, datasets) -> None:
        """Not merely the same length: the same numbers. A series shifted by
        one bar would pass an index-length check and still be wrong."""
        train = datasets[LARGE].training_set()
        assert np.array_equal(train.series.to_numpy(), train.y.to_numpy())

    def test_the_contract_refuses_a_series_longer_than_the_budget(self, datasets) -> None:
        """The failure that happened, as a constructor call: it must not be
        possible to build a TrainingSet whose series reaches past its rows."""
        dataset = datasets[SMALL]
        budget = dataset.partition.budget
        with pytest.raises(ValueError, match="does not grant"):
            TrainingSet(
                X=dataset.X.loc[budget],
                y=dataset.y.loc[budget],
                series=dataset.series.loc[dataset.series.index <= budget.max()],
                close=dataset.close.loc[budget],
                feature_bar=dataset.feature_bar.loc[budget],
            )

    def test_the_contract_refuses_a_misaligned_close(self, datasets) -> None:
        dataset = datasets[SMALL]
        budget = dataset.partition.budget
        with pytest.raises(ValueError, match="training close"):
            TrainingSet(
                X=dataset.X.loc[budget],
                y=dataset.y.loc[budget],
                series=dataset.series.loc[budget],
                close=dataset.close.loc[dataset.close.index <= budget.max()],
                feature_bar=dataset.feature_bar.loc[budget],
            )


class TestEveryModelEstimatesOnTheBudget:
    def test_the_state_space_family_is_under_test(self) -> None:
        """Guard the guard: an empty parametrisation would pass vacuously."""
        pytest.importorskip("statsmodels")
        assert len(state_space_models()) >= 5

    @pytest.mark.parametrize("model_id", state_space_models())
    def test_state_space_estimation_uses_exactly_the_budget(self, datasets, model_id) -> None:
        """The direct measurement: how many observations the estimator saw."""
        for n in (SMALL, LARGE):
            model = registry.build(model_id).fit(datasets[n].training_set())
            assert int(model._result.nobs) == n, (
                f"{model_id} estimated on {int(model._result.nobs)} observations "
                f"at a budget of {n}"
            )

    @pytest.mark.parametrize(
        "model_id",
        [m for m in runnable(exclude_family=Family.DEEP) if m != "naive_last_value"],
    )
    def test_the_forecast_depends_on_the_budget(self, datasets, model_id) -> None:
        """The generic guard, and the symptom that exposed the bug: a model
        whose holdout forecasts are identical at two different budgets is not
        estimating on its budget. `naive_last_value` is exempt because it
        estimates nothing; the deep family is left out only for runtime, and
        builds its windows from `X`, which was always bounded."""
        forecasts = []
        for n in (SMALL, LARGE):
            dataset = datasets[n]
            model = registry.build(model_id).fit(dataset.training_set())
            if model.needs_calibration:
                model.calibrate(dataset.development_context())
            forecasts.append(model.predict(dataset.evaluation_context()).point)
        assert not np.array_equal(forecasts[0], forecasts[1]), (
            f"{model_id} produced identical forecasts at budgets {SMALL} and {LARGE}"
        )
