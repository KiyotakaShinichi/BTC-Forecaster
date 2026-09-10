"""The runner: deterministic execution, canonical output, and verification.

Three properties, each pinned:

* **The worker count changes nothing but the time.** One worker and two produce
  byte-identical canonical files and the same result digest.
* **The canonical files carry no run identity.** No timestamps, no timings, no
  run ids -- those live in ``run_info.json``, outside the digest.
* **A written run can be checked.** ``verify_run`` recomputes every hash from
  disk and names the file that was touched, or that is not there.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from btc_forecaster.data.snapshot import MarketSnapshot
from btc_forecaster.features.spec import default_specs
from btc_forecaster.research.walk_forward.__main__ import (
    EXIT_INPUT_CHANGED,
    EXIT_INTEGRITY,
    EXIT_OK,
    main,
)
from btc_forecaster.research.walk_forward.config import (
    PREPROCESSING_VERSION,
    WalkForwardConfig,
    WindowSpec,
)
from btc_forecaster.research.walk_forward.manifest import input_manifest
from btc_forecaster.research.walk_forward.runner import (
    CANONICAL_FILES,
    attack_plan,
    canonical_contents,
    job_specs,
    run_benchmark,
    verify_run,
    write_run,
)
from btc_forecaster.research.walk_forward.stability import DECISIONS, LEAKAGE_PASSED
from btc_forecaster.research.walk_forward.targets import TARGET_DEFINITION
from btc_forecaster.research.walk_forward.worlds import build

FRAME = build("AUTOCORRELATION").frame
CONFIG = WalkForwardConfig(
    models=("naive_last_value", "random_walk_drift", "ar_p"),
    horizons=(1, 3),
    windows=(WindowSpec("rolling", 150), WindowSpec("expanding")),
    n_refits=3,
    bootstrap_resamples=100,
    minimum_paired_origins=30,
)
MANIFEST = input_manifest(
    MarketSnapshot.build(
        FRAME, ticker="SYNTHETIC-AUTOCORRELATION", provider="synthetic",
        retrieved_at=pd.Timestamp("2000-01-01", tz="UTC"),
    ),
    file_bytes=None,
    preprocessing_version=PREPROCESSING_VERSION,
    feature_specs=tuple(s.name for s in default_specs()),
    target=TARGET_DEFINITION,
)


@pytest.fixture(scope="module")
def one_worker():
    return run_benchmark(FRAME, CONFIG, workers=1)


@pytest.fixture(scope="module")
def two_workers():
    return run_benchmark(FRAME, CONFIG, workers=2)


class TestThePlan:
    def test_every_model_horizon_and_window_is_a_job(self) -> None:
        assert len(job_specs(CONFIG)) == 3 * 2 * 2

    def test_every_model_is_attacked_at_both_ends_of_the_horizon_range(self) -> None:
        plan = attack_plan(CONFIG)
        assert {(m, h) for m, h, _, _ in plan} == {(m, h) for m in CONFIG.models for h in (1, 3)}
        assert {w for _, _, w, _ in plan} == {"rolling-150"}
        assert {f for _, _, _, f in plan} == {CONFIG.n_refits - 1}


class TestTheWorkerCountChangesNothingButTheTime:
    def test_canonical_files_are_byte_identical(self, one_worker, two_workers) -> None:
        a = canonical_contents(one_worker, MANIFEST)
        b = canonical_contents(two_workers, MANIFEST)
        assert set(a) == set(CANONICAL_FILES)
        for name in CANONICAL_FILES:
            assert a[name] == b[name], name

    def test_the_decision_and_leakage_agree(self, one_worker, two_workers) -> None:
        assert one_worker.decision == two_workers.decision
        assert set(one_worker.leakage.values()) == {LEAKAGE_PASSED}


class TestTheWrittenRun:
    @pytest.fixture(scope="class")
    def written(self, one_worker, tmp_path_factory):
        first = tmp_path_factory.mktemp("run1")
        second = tmp_path_factory.mktemp("run2")
        return (
            first, write_run(one_worker, first, input_manifest=MANIFEST),
            second, write_run(one_worker, second, input_manifest=MANIFEST),
        )

    def test_two_writes_are_byte_identical(self, written) -> None:
        first, m1, second, m2 = written
        assert m1 == m2
        for name in (*[n for n in CANONICAL_FILES if n != "predictions.csv"], "predictions.csv.gz", "manifest.json"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def test_predictions_are_stored_compressed_and_hashed_uncompressed(self, written, one_worker) -> None:
        first, manifest, _, _ = written
        raw = gzip.decompress((first / "predictions.csv.gz").read_bytes())
        assert raw == canonical_contents(one_worker, MANIFEST)["predictions.csv"]
        assert "predictions.csv" in manifest["digest_covers"]

    def test_no_canonical_file_carries_run_identity(self, written) -> None:
        first, _, _, _ = written
        for name in CANONICAL_FILES:
            path = first / ("predictions.csv.gz" if name == "predictions.csv" else name)
            text = gzip.decompress(path.read_bytes()).decode() if name == "predictions.csv" else path.read_text()
            for token in ("generated_at", "seconds", "run_id", "hostname"):
                assert token not in text, (name, token)

    def test_timings_live_outside_the_digest(self, written) -> None:
        first, manifest, _, _ = written
        info = json.loads((first / "run_info.json").read_text())
        assert "timings" in info and "generated_at" in info
        assert "run_info.json" in manifest["digest_excludes"]

    def test_the_manifest_promotes_nothing(self, written) -> None:
        _, manifest, _, _ = written
        assert manifest["promoted_models"] == [] and manifest["live_trading_enabled"] is False
        assert manifest["decision"] in DECISIONS

    def test_verification_passes_untouched(self, written) -> None:
        first, _, _, _ = written
        assert verify_run(first)["verified"]

    def test_verification_names_a_tampered_file(self, written) -> None:
        _, _, second, _ = written
        metrics = second / "metrics.json"
        metrics.write_text(metrics.read_text().replace("0", "1", 1))
        result = verify_run(second)
        assert not result["verified"] and result["mismatched_files"] == ["metrics.json"]

    def test_tampered_predictions_are_caught_through_the_compression(self, tmp_path, one_worker) -> None:
        write_run(one_worker, tmp_path, input_manifest=MANIFEST)
        raw = gzip.decompress((tmp_path / "predictions.csv.gz").read_bytes()).replace(b"True", b"False", 1)
        (tmp_path / "predictions.csv.gz").write_bytes(gzip.compress(raw, mtime=0))
        assert verify_run(tmp_path)["mismatched_files"] == ["predictions.csv"]

    def test_an_absent_file_is_named_and_the_rest_are_still_checked(self, tmp_path, one_worker) -> None:
        write_run(one_worker, tmp_path, input_manifest=MANIFEST)
        (tmp_path / "predictions.csv.gz").unlink()
        result = verify_run(tmp_path)
        assert result["absent_files"] == ["predictions.csv"]
        assert result["mismatched_files"] == [] and result["result_digest_recomputed"] is None
        assert not result["verified"]
        metrics = tmp_path / "metrics.json"
        metrics.write_text(metrics.read_text().replace("0", "1", 1))
        assert verify_run(tmp_path)["mismatched_files"] == ["metrics.json"]


class TestTheCommand:
    def test_config_prints_the_digest(self, capsys) -> None:
        assert main(["config"]) == EXIT_OK
        assert WalkForwardConfig().digest() in capsys.readouterr().out

    def test_a_smoke_run_on_a_world_writes_a_verifiable_result(self, tmp_path, capsys) -> None:
        assert main(["run", "--world", "NOISE", "--smoke", "--output", str(tmp_path)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "decision" in out and "digest" in out
        assert main(["verify", str(tmp_path)]) == EXIT_OK
        assert (tmp_path / "README.md").read_text().startswith("# A7 walk-forward robustness")

    def test_an_unexpected_input_is_refused(self, tmp_path, capsys) -> None:
        code = main(["run", "--world", "NOISE", "--smoke", "--output", str(tmp_path), "--expect-input", "0" * 64])
        assert code == EXIT_INPUT_CHANGED
        assert "different experiment" in capsys.readouterr().err
        assert not (tmp_path / "manifest.json").exists()

    def test_verifying_nothing_is_an_integrity_failure(self, tmp_path) -> None:
        assert main(["verify", str(tmp_path / "absent")]) == EXIT_INTEGRITY

    def test_a_run_without_its_predictions_is_not_verified_and_says_why(self, tmp_path, capsys, one_worker) -> None:
        write_run(one_worker, tmp_path, input_manifest=MANIFEST)
        (tmp_path / "predictions.csv.gz").unlink()
        assert main(["verify", str(tmp_path)]) == EXIT_INTEGRITY
        captured = capsys.readouterr()
        assert json.loads(captured.out)["absent_files"] == ["predictions.csv"]
        assert "predictions.csv.gz is not in" in captured.err and "Regenerate" in captured.err
