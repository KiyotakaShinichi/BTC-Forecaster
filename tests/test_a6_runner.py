"""The benchmark runner: determinism, telemetry, and the promotion it cannot do.

Three properties are tested here because they are the ones that make the output
trustworthy rather than merely present:

**Failures are recorded.** A model that raises must appear in the table as
FAILED with its exception, and the other thirty-nine must still run. The failure
mode this prevents is the quiet one -- a benchmark that looks better than it is
because the models that broke are simply not in it.

**Nothing is promoted.** There is no `promoted` field to set, every result
carries EXPLORATORY, and `assert_nothing_promoted` refuses a manifest that says
otherwise. Cheap to check, so it is checked.

**The manifest is written last.** Its presence is the claim that the run
finished; a crash halfway through must leave artifacts without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.contracts import (
    EXPLORATORY,
    EvaluationContext,
    Family,
    ModelStatus,
    ResourceClass,
    TrainingSet,
    ZooModel,
)
from btc_forecaster.research.metrics import SMAPE_IS_MEANINGFUL
from btc_forecaster.research.model_zoo import (
    SAMPLE_EFFICIENCY_BUDGETS,
    SAMPLE_EFFICIENCY_MODELS,
    build_parser,
    main,
)
from btc_forecaster.research.partition import PartitionSpec
from btc_forecaster.research.registry import ZooRegistration, register
from btc_forecaster.research.runner import (
    BENCHMARK_KIND,
    assert_nothing_promoted,
    build_manifest,
    run_benchmark,
    write_run,
)
from btc_forecaster.testing import synthetic_market_frame

#: A cheap subset. The full zoo takes three minutes; the tests must not.
FAST_MODELS = ["naive_last_value", "random_walk_drift", "ridge", "decision_tree"]


@pytest.fixture(scope="module")
def frame():
    return synthetic_market_frame(periods=500, kind="ar1", seed=31)


@pytest.fixture(scope="module")
def result(frame):
    return run_benchmark(frame, spec=PartitionSpec(train_rows=150), model_ids=FAST_MODELS)


class TestTheRunIsDeterministic:
    def test_two_runs_agree_exactly(self, frame) -> None:
        """A benchmark that cannot be re-run to the same number is not evidence."""
        spec = PartitionSpec(train_rows=150)
        first = run_benchmark(frame, spec=spec, model_ids=FAST_MODELS)
        second = run_benchmark(frame, spec=spec, model_ids=FAST_MODELS)
        for model_id in FAST_MODELS:
            assert np.array_equal(
                first.forecasts[model_id].point, second.forecasts[model_id].point
            ), model_id

    def test_every_model_records_the_rows_it_saw(self, result) -> None:
        """The fingerprint is what makes 'the same training rows' checkable
        rather than assumed."""
        fingerprints = {
            outcome.train_fingerprint for outcome in result.succeeded()
        }
        assert len(fingerprints) == 1
        assert next(iter(fingerprints))

    def test_the_data_is_fingerprinted(self, result) -> None:
        assert len(result.data_fingerprint) == 64


class TestFailuresAreRecordedNotSwallowed:
    def test_a_raising_model_is_recorded_and_the_run_continues(self, frame) -> None:
        class Exploding(ZooModel):
            model_id = "exploding_test_model"
            family = Family.BASELINE

            def _fit(self, train: TrainingSet) -> None:
                raise ValueError("deliberate failure")

            def _predict_point(self, context: EvaluationContext) -> np.ndarray:
                return np.zeros(len(context))

        register(
            ZooRegistration(
                model_id="exploding_test_model",
                factory=Exploding,
                family=Family.BASELINE,
                resource_class=ResourceClass.TRIVIAL,
                description="raises on purpose",
            )
        )
        result = run_benchmark(
            frame,
            spec=PartitionSpec(train_rows=150),
            model_ids=["naive_last_value", "exploding_test_model", "ridge"],
        )
        failed = [o for o in result.outcomes if o.status is ModelStatus.FAILED]
        assert [o.model_id for o in failed] == ["exploding_test_model"]
        assert failed[0].failure["exception"] == "ValueError"
        assert "deliberate failure" in failed[0].failure["message"]
        assert failed[0].failure["traceback"]
        # The other two still ran.
        assert len(result.succeeded()) == 2

    def test_an_unsuitable_model_carries_its_reason(self, frame) -> None:
        result = run_benchmark(
            frame, spec=PartitionSpec(train_rows=150), model_ids=["sarima"]
        )
        outcome = result.outcomes[0]
        assert outcome.status is ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB
        assert "ACF(7)" in outcome.failure["reason"]

    def test_every_attempted_model_appears_somewhere(self, result) -> None:
        """The invariant: no silent omissions."""
        listed = {model_id for ids in result.by_status().values() for model_id in ids}
        assert listed == set(FAST_MODELS)

    def test_the_status_vocabulary_is_closed(self, result) -> None:
        allowed = {status.value for status in ModelStatus}
        assert set(result.by_status()) <= allowed


class TestNothingIsPromoted:
    def test_the_manifest_says_so_in_four_places(self, result) -> None:
        manifest = build_manifest(result)
        assert manifest["scientific_status"] == EXPLORATORY
        assert manifest["promoted_models"] == []
        assert manifest["live_trading_enabled"] is False
        assert "does not supersede A2" in manifest["warning"]

    def test_every_model_row_is_exploratory(self, result) -> None:
        manifest = build_manifest(result)
        for model in manifest["models"]:
            assert model["scientific_status"] == EXPLORATORY

    def test_the_assertion_catches_a_promoted_model(self, result) -> None:
        manifest = build_manifest(result)
        manifest["promoted_models"] = ["ridge"]
        with pytest.raises(AssertionError, match="must not"):
            assert_nothing_promoted(manifest)

    def test_the_assertion_catches_enabled_trading(self, result) -> None:
        manifest = build_manifest(result)
        manifest["live_trading_enabled"] = True
        with pytest.raises(AssertionError, match="live trading"):
            assert_nothing_promoted(manifest)

    def test_the_assertion_catches_a_downgraded_status(self, result) -> None:
        manifest = build_manifest(result)
        manifest["models"][0]["scientific_status"] = "PROMOTED"
        with pytest.raises(AssertionError, match="not EXPLORATORY"):
            assert_nothing_promoted(manifest)

    def test_best_is_reported_per_family_never_globally(self, result) -> None:
        """The best of forty draws from a distribution centred on nothing is
        still the best of forty."""
        best = result.best_by_family()
        assert best
        for family, entry in best.items():
            assert entry["scientific_status"] == EXPLORATORY
            assert "not a promotion" in entry["note"], family
        manifest = build_manifest(result)
        assert "global_winner" not in manifest
        assert "best_model" not in manifest


class TestTheArtifactsAreHonest:
    def test_the_manifest_is_written_last(self, result, tmp_path: Path) -> None:
        """Its presence is the claim that the run finished."""
        manifest_path = write_run(result, tmp_path)
        assert manifest_path.name == "manifest.json"
        written = sorted(p.name for p in tmp_path.iterdir())
        assert {"manifest.json", "predictions.csv", "results.csv"} <= set(written)
        newest = max(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
        assert newest.name == "manifest.json"

    def test_the_manifest_records_the_benchmark_kind(self, result) -> None:
        assert build_manifest(result)["benchmark"] == BENCHMARK_KIND
        assert "RESOURCE_CONSTRAINED" in BENCHMARK_KIND

    def test_smape_is_kept_but_flagged_as_not_meaningful(self, result) -> None:
        """The target is a signed log return that crosses zero, so sMAPE
        saturates near 200% for everything including the naive baseline. Kept
        for inspection, excluded from the ranking, with the reason recorded."""
        manifest = build_manifest(result)
        assert SMAPE_IS_MEANINGFUL is False
        assert "saturates" in manifest["metric_notes"]["smape"]
        scored = next(m for m in manifest["models"] if m["scores"])
        assert scored["scores"]["point"]["smape_is_meaningful"] is False
        assert "smape" not in result.results_frame().columns

    def test_the_directional_null_is_not_a_coin(self, result) -> None:
        """A2 tested against 0.5 once. On a series with a 53% up-day base rate,
        always saying 'up' beats a coin and contains no information."""
        manifest = build_manifest(result)
        assert "0.5" in manifest["metric_notes"]["directional_null"]
        scored = next(m for m in manifest["models"] if m["scores"])
        direction = scored["scores"]["direction"]
        assert "train_constant_baseline" in direction
        assert "base_rate" in direction

    def test_absent_capabilities_stay_absent_in_the_manifest(self, result) -> None:
        """A blank is a fact; a filled-in default is a fabrication."""
        manifest = build_manifest(result)
        ridge = next(m for m in manifest["models"] if m["model_id"] == "ridge")
        assert ridge["scores"]["probabilistic"] is None
        assert ridge["scores"]["variance"] is None

    def test_predictions_are_written_with_their_target_bars(
        self, result, tmp_path: Path
    ) -> None:
        write_run(result, tmp_path)
        import pandas as pd

        predictions = pd.read_csv(tmp_path / "predictions.csv")
        assert "target_bar" in predictions.columns
        assert "actual" in predictions.columns
        for model_id in FAST_MODELS:
            assert model_id in predictions.columns

    def test_the_registry_travels_with_the_run(self, result, tmp_path: Path) -> None:
        """So a run can be read years later without the code that made it."""
        write_run(result, tmp_path)
        entries = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
        assert len(entries) == len(registry.all_registrations())


class TestTheCommandLine:
    def test_list_prints_from_the_registry(self, capsys) -> None:
        """One source of truth. There is no second model list in the CLI."""
        assert main(["--list"]) == 0
        printed = capsys.readouterr().out
        for model_id in ("naive_last_value", "transformer", "sarima"):
            assert model_id in printed
        assert "UNSUITABLE_FOR_CONSTRAINED_LAB" in printed

    def test_a_missing_snapshot_explains_itself(self, capsys, tmp_path: Path) -> None:
        code = main(["--data", str(tmp_path / "absent.csv")])
        assert code == 2
        assert "not committed" in capsys.readouterr().err

    def test_the_sample_efficiency_subset_is_curated(self) -> None:
        """Phase 14 is explicit that the whole zoo must not run at every budget."""
        assert len(SAMPLE_EFFICIENCY_MODELS) < len(registry.model_ids()) / 4
        assert SAMPLE_EFFICIENCY_BUDGETS == (250, 500, 1000)
        families = {registry.get(m).family for m in SAMPLE_EFFICIENCY_MODELS}
        assert len(families) >= 5

    def test_the_parser_defaults_to_one_thousand_rows(self) -> None:
        assert build_parser().parse_args([]).train_rows == 1000
