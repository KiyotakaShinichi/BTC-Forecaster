"""Save, reload, and prove the forecasts are identical -- plus the cards.

A benchmark that cannot be reloaded is a benchmark whose results expire with the
process that produced them. Every model declares `Capability.SERIALIZE`, and
these tests make that claim checkable across all six families.

Bit-identical, not close. A tolerance would hide the failure this exists for: a
reloaded model that lost its scaler, its calibration block or its frozen
parameters does not produce slightly different numbers, it produces plausible
ones. Every family round-trips exactly, so exactness is the honest bar.

The card tests check that nothing is hand-written. Phase 26 forbids authoring
individual results, and the reason is that a hand-written card states what
somebody believed when they wrote it -- and stops being true silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_forecaster.research import registry
from btc_forecaster.research.cards import SHARED_LIMITATIONS, build_card
from btc_forecaster.research.contracts import EXPLORATORY, Capability
from btc_forecaster.research.partition import PartitionSpec
from btc_forecaster.research.runner import analyse, build_manifest, run_benchmark, write_run
from btc_forecaster.research.serialization import (
    Artifact,
    ArtifactIntegrityError,
    deserialize,
    round_trip_is_exact,
    serialize,
)
from btc_forecaster.testing import synthetic_market_frame


def available(*model_ids: str) -> list[str]:
    """Filter to the models this environment can actually build.

    The quant CI job installs `.[dev]` -- numpy, pandas, scipy and statsmodels,
    but not scikit-learn, arch or xgboost. Hard-coding a model list would make
    these tests either fail there or be skipped wholesale, and the second is
    worse: the leakage suite silently not running is exactly the situation it
    exists to prevent. Filtering keeps whatever is present under test.
    """
    return [m for m in model_ids if registry.get(m).is_available()]


#: One per family, chosen so every serialization mechanism is exercised:
#: a constant, a statsmodels result, an arch parameter vector, an sklearn
#: estimator, a multi-model quantile wrapper, and a numpy network.
FAMILY_SAMPLE = available(
    "naive_last_value",
    "arima",
    "garch_11",
    "ridge",
    "knn",
    "quantile_gbr",
    "conformal_ridge",
    "gru",
)


@pytest.fixture(scope="module")
def result():
    frame = synthetic_market_frame(periods=420, kind="ar1", seed=53)
    return run_benchmark(frame, spec=PartitionSpec(train_rows=150), model_ids=FAMILY_SAMPLE)


class TestEveryFamilyRoundTrips:
    @pytest.mark.parametrize("model_id", FAMILY_SAMPLE)
    def test_the_forecasts_are_bit_identical(self, result, model_id) -> None:
        model = result.models.get(model_id)
        if model is None:
            pytest.skip(f"{model_id} did not run in this environment")
        assert round_trip_is_exact(model, result.dataset.evaluation_context())

    @pytest.mark.parametrize("model_id", FAMILY_SAMPLE)
    def test_the_artifact_records_its_own_hash_and_size(self, result, model_id) -> None:
        model = result.models.get(model_id)
        if model is None:
            pytest.skip(f"{model_id} did not run")
        artifact = serialize(model)
        assert len(artifact.sha256) == 64
        assert artifact.bytes_written == len(artifact.payload)
        assert artifact.as_dict()["model_id"] == model_id

    def test_the_environment_travels_with_the_artifact(self, result) -> None:
        """A pickle is not a data format. What wrote it has to be recorded, or
        a reload failure years from now is unexplainable."""
        pytest.importorskip("sklearn")
        environment = serialize(result.models["ridge"]).as_dict()["environment"]
        assert "python" in environment
        assert "numpy" in environment

    def test_a_tampered_artifact_is_refused(self, result) -> None:
        """Unpickling executes. The hash is the only thing between a swapped
        artifact and arbitrary code, and checking it costs a microsecond."""
        pytest.importorskip("sklearn")
        artifact = serialize(result.models["ridge"])
        tampered = Artifact(
            model_id=artifact.model_id,
            payload=artifact.payload + b"\x00",
            sha256=artifact.sha256,
            bytes_written=artifact.bytes_written,
            created_at=artifact.created_at,
            environment=artifact.environment,
        )
        with pytest.raises(ArtifactIntegrityError, match="hashes to"):
            deserialize(tampered)

    def test_the_deep_family_survives_losing_its_tape(self, result) -> None:
        """A Tensor holds a backward closure, and a lambda cannot be pickled.
        Dropping the tape on serialization is what makes the deep models
        saveable at all -- and the reloaded parameters must still predict."""
        gru = result.models["gru"]
        reloaded = deserialize(serialize(gru))
        context = result.dataset.evaluation_context()
        assert np.array_equal(gru.predict(context).point, reloaded.predict(context).point)

    @pytest.mark.parametrize("model_id", FAMILY_SAMPLE)
    def test_the_capability_is_declared(self, result, model_id) -> None:
        model = result.models.get(model_id)
        if model is None:
            pytest.skip(f"{model_id} did not run")
        assert model.supports(Capability.SERIALIZE)


class TestTheRunWritesItsArtifacts:
    def test_a_written_run_carries_hashes_and_round_trip_flags(self, result, tmp_path) -> None:
        write_run(result, tmp_path)
        import json

        artifacts = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
        assert artifacts
        for model_id, entry in artifacts.items():
            assert entry.get("round_trip_exact") is True, model_id
            assert len(entry["sha256"]) == 64

    def test_the_analysis_is_written_beside_the_results(self, result, tmp_path) -> None:
        write_run(result, tmp_path)
        import json

        analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
        assert set(analysis) >= {"comparison", "stability", "diversity", "diagnostics"}

    def test_the_manifest_is_still_written_last(self, result, tmp_path) -> None:
        write_run(result, tmp_path)
        newest = max(
            (p for p in tmp_path.rglob("*") if p.is_file()), key=lambda p: p.stat().st_mtime
        )
        assert newest.name == "manifest.json"


class TestModelCardsAreGenerated:
    def test_one_card_per_model_including_those_that_did_not_run(
        self, result, tmp_path
    ) -> None:
        """A model absent from the card directory is a model absent from the
        record, which is what the status vocabulary exists to prevent."""
        write_run(result, tmp_path)
        cards = {p.stem for p in (tmp_path / "cards").glob("*.md")}
        assert cards == {outcome.model_id for outcome in result.outcomes}

    def test_a_card_is_derived_not_authored(self, result) -> None:
        """Every number on it comes from the manifest."""
        pytest.importorskip("sklearn")
        manifest = build_manifest(result)
        model = next(m for m in manifest["models"] if m["model_id"] == "ridge")
        registration = next(
            r for r in manifest["registry_entries"] if r["model_id"] == "ridge"
        )
        card = build_card(model, registration=registration, naive_mae=manifest["naive_mae"])
        assert f"{model['scores']['point']['mae']:.6f}" in card

    def test_every_card_states_the_limitations(self, result, tmp_path) -> None:
        write_run(result, tmp_path)
        for path in (tmp_path / "cards").glob("*.md"):
            text = path.read_text(encoding="utf-8")
            assert SHARED_LIMITATIONS[0][:40] in text, path.name
            assert "1,000 training rows" in text

    def test_every_card_is_exploratory_and_none_recommends_trading(
        self, result, tmp_path
    ) -> None:
        write_run(result, tmp_path)
        for path in (tmp_path / "cards").glob("*.md"):
            text = path.read_text(encoding="utf-8")
            assert f"**Scientific status** {EXPLORATORY}" in text, path.name
            assert "Nothing in this card is a recommendation to trade" in text, path.name
            # The only permitted mention of promotion is the denial of it.
            assert "**Status** PROMOTED" not in text, path.name

    def test_every_card_is_pure_ascii(self, result, tmp_path) -> None:
        """These are read back by whatever tooling a reader has. A stray
        typographic separator turns into a replacement character on any console
        that is not UTF-8, which is most of them on this platform."""
        write_run(result, tmp_path)
        for path in (tmp_path / "cards").glob("*.md"):
            text = path.read_text(encoding="utf-8")
            offending = sorted({c for c in text if ord(c) > 127})
            assert not offending, f"{path.name}: {offending}"

    def test_a_card_says_when_its_model_is_worse_than_the_baseline(self, result) -> None:
        """In the same place it would have said the opposite."""
        manifest = build_manifest(result)
        analyses = analyse(result)
        rows = {r["model_id"]: r for r in analyses["comparison"].get("results", [])}
        worse = [
            r
            for r in rows.values()
            if r.get("significant_after_bh") and (r.get("mean_loss_difference") or 0) > 0
        ]
        if not worse:
            pytest.skip("no model was significantly worse in this sample")
        model_id = worse[0]["model_id"]
        model = next(m for m in manifest["models"] if m["model_id"] == model_id)
        registration = next(
            r for r in manifest["registry_entries"] if r["model_id"] == model_id
        )
        card = build_card(model, registration=registration, comparison=rows[model_id])
        assert "significantly **worse**" in card

    def test_a_card_names_what_the_model_declined_to_produce(self, result) -> None:
        """A blank cell in the table is a fact about the model, and a card that
        omits it invites the reader to assume the number was not computed."""
        point_only = next(
            m for m in ("ridge", "ar_p", "naive_last_value") if m in result.models
        )
        manifest = build_manifest(result)
        model = next(m for m in manifest["models"] if m["model_id"] == point_only)
        registration = next(
            r for r in manifest["registry_entries"] if r["model_id"] == point_only
        )
        card = build_card(model, registration=registration)
        assert "does not" in card
        assert "not declared" in card
