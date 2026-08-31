"""Origin generation, chunked/resumable replay, catalog, and extension.

The tests that matter here are the ones about failure: an interrupted run must
not look complete, and an extension under changed conditions must refuse rather
than silently rebuild history.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from market_intelligence.errors import ReplayIntegrityError
from market_intelligence.feature_matrix import (
    MATRIX_COLUMNS,
    PART_PREFIX,
    PROGRESS_FILENAME,
    HistoricalFeatureMatrixBuilder,
    membership_hash,
    source_fingerprint,
)
from market_intelligence.historical import HistoricalDatasetService
from market_intelligence.origins import (
    OriginFrequency,
    OriginScheduleError,
    generate_origins,
    infer_frequency,
    validate_origin_schedule,
)
from market_intelligence.replay_dataset import ReplayMode
from market_intelligence.storage import IntelligenceStore
from tests.test_intelligence_replay_engine import CONFIG, PROVIDERS, golden_fixture

BASE = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[IntelligenceStore]:
    documents, events = golden_fixture()
    store = IntelligenceStore(tmp_path / "intelligence.duckdb")
    store.put_documents(documents)
    store.put_signals(events)
    yield store
    store.close()


def hourly(hours_back: int, hours_forward: int = 0) -> list[datetime]:
    return generate_origins(
        BASE - timedelta(hours=hours_back), BASE + timedelta(hours=hours_forward), OriginFrequency.HOURLY
    )


class TestOriginGenerator:
    def test_hourly_is_inclusive_at_both_ends(self) -> None:
        origins = generate_origins(BASE, BASE + timedelta(hours=3), OriginFrequency.HOURLY)
        assert origins[0] == BASE
        assert origins[-1] == BASE + timedelta(hours=3)
        assert len(origins) == 4

    @pytest.mark.parametrize(
        ("frequency", "expected_step"),
        [
            (OriginFrequency.HOURLY, timedelta(hours=1)),
            (OriginFrequency.FOUR_HOURLY, timedelta(hours=4)),
            (OriginFrequency.DAILY, timedelta(days=1)),
        ],
    )
    def test_every_cadence_steps_correctly(
        self, frequency: OriginFrequency, expected_step: timedelta
    ) -> None:
        origins = generate_origins(BASE, BASE + expected_step * 3, frequency)
        gaps = {later - earlier for earlier, later in zip(origins, origins[1:], strict=False)}
        assert gaps == {expected_step}

    def test_an_end_off_the_cadence_is_excluded(self) -> None:
        origins = generate_origins(BASE, BASE + timedelta(hours=3, minutes=30), OriginFrequency.HOURLY)
        assert origins[-1] == BASE + timedelta(hours=3)

    def test_a_naive_datetime_is_rejected_not_assumed(self) -> None:
        """Guessing would make the dataset depend on the machine that built it."""
        with pytest.raises(OriginScheduleError, match="timezone-aware"):
            generate_origins(datetime(2025, 6, 1), BASE, OriginFrequency.HOURLY)

    def test_a_non_utc_input_is_converted_not_rejected(self) -> None:
        tokyo = BASE.astimezone(timezone(timedelta(hours=9)))
        origins = generate_origins(tokyo, tokyo + timedelta(hours=2), OriginFrequency.HOURLY)
        assert all(origin.tzinfo == timezone.utc for origin in origins)
        assert origins[0] == BASE

    def test_generation_is_deterministic(self) -> None:
        assert hourly(24) == hourly(24)

    def test_origins_are_sorted_and_unique(self) -> None:
        origins = hourly(48)
        assert origins == sorted(origins)
        assert len(set(origins)) == len(origins)

    def test_end_before_start_is_an_error(self) -> None:
        with pytest.raises(OriginScheduleError, match="precedes start"):
            generate_origins(BASE, BASE - timedelta(hours=1), OriginFrequency.HOURLY)

    def test_limit_truncates_from_the_start(self) -> None:
        """A truncated schedule must be a prefix, or extension is ill-defined."""
        full = generate_origins(BASE, BASE + timedelta(hours=10), OriginFrequency.HOURLY)
        limited = generate_origins(BASE, BASE + timedelta(hours=10), OriginFrequency.HOURLY, limit=4)
        assert limited == full[:4]

    def test_validate_rejects_unsorted_and_duplicate_schedules(self) -> None:
        with pytest.raises(OriginScheduleError, match="sorted"):
            validate_origin_schedule([BASE + timedelta(hours=1), BASE])
        with pytest.raises(OriginScheduleError, match="unique"):
            validate_origin_schedule([BASE, BASE])
        with pytest.raises(OriginScheduleError, match="empty"):
            validate_origin_schedule([])

    def test_frequency_can_be_inferred_and_irregularity_detected(self) -> None:
        assert infer_frequency(hourly(5)) is OriginFrequency.HOURLY
        assert infer_frequency([BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=3)]) is None
        assert infer_frequency([BASE]) is None


class TestChunkedBuild:
    def test_rows_have_one_entry_per_origin_with_every_column(self, store: IntelligenceStore, tmp_path: Path) -> None:
        origins = hourly(12)
        builder = HistoricalFeatureMatrixBuilder(store, chunk_size=5)
        rows, snapshot_ids, chunks = builder.build_chunks(
            origins, tmp_path / "work", PROVIDERS, CONFIG
        )

        assert len(rows) == len(origins) == len(snapshot_ids)
        assert set(rows[0]) == set(MATRIX_COLUMNS)
        assert [chunk.row_count for chunk in chunks] == [5, 5, 3]

    def test_chunking_does_not_change_the_rows(self, store: IntelligenceStore, tmp_path: Path) -> None:
        origins = hourly(12)
        one_chunk, _, _ = HistoricalFeatureMatrixBuilder(store, chunk_size=100).build_chunks(
            origins, tmp_path / "a", PROVIDERS, CONFIG
        )
        many_chunks, _, _ = HistoricalFeatureMatrixBuilder(store, chunk_size=2).build_chunks(
            origins, tmp_path / "b", PROVIDERS, CONFIG
        )
        assert one_chunk == many_chunks

    def test_reference_and_optimized_modes_produce_identical_rows(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        origins = hourly(24)
        builder = HistoricalFeatureMatrixBuilder(store, chunk_size=7)
        reference, reference_ids, _ = builder.build_chunks(
            origins, tmp_path / "ref", PROVIDERS, CONFIG, mode=ReplayMode.REFERENCE
        )
        optimized, optimized_ids, _ = builder.build_chunks(
            origins, tmp_path / "opt", PROVIDERS, CONFIG, mode=ReplayMode.OPTIMIZED
        )
        assert reference == optimized
        assert reference_ids == optimized_ids

    def test_ordering_is_deterministic_with_no_duplicate_origins(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        origins = hourly(20)
        rows, _, _ = HistoricalFeatureMatrixBuilder(store, chunk_size=6).build_chunks(
            origins, tmp_path / "work", PROVIDERS, CONFIG
        )
        emitted = [row["forecast_origin"] for row in rows]
        assert emitted == [origin.isoformat() for origin in origins]
        assert len(set(emitted)) == len(emitted)

    def test_a_zero_chunk_size_is_rejected(self, store: IntelligenceStore) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            HistoricalFeatureMatrixBuilder(store, chunk_size=0)


class TestResumption:
    def test_completed_chunks_are_reused_not_recomputed(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        origins = hourly(12)
        work = tmp_path / "work"
        builder = HistoricalFeatureMatrixBuilder(store, chunk_size=5)
        first, _, _ = builder.build_chunks(origins, work, PROVIDERS, CONFIG)

        # Simulate an interruption: drop the last part, keep the earlier ones.
        parts = sorted(work.glob(f"{PART_PREFIX}*.json"))
        assert len(parts) == 3
        parts[-1].unlink()

        second, _, chunks = builder.build_chunks(origins, work, PROVIDERS, CONFIG)
        assert second == first
        assert len(chunks) == 3

    def test_partial_output_is_not_mistaken_for_a_dataset(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        """The manifest is the completion marker. Parts alone are scratch."""
        origins = hourly(12)
        work = tmp_path / "work"
        HistoricalFeatureMatrixBuilder(store, chunk_size=5).build_chunks(
            origins, work, PROVIDERS, CONFIG
        )
        progress = json.loads((work / PROGRESS_FILENAME).read_text(encoding="utf-8"))
        assert progress["complete"] is False

    def test_a_different_request_discards_the_previous_parts(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        """Resuming into another dataset's parts would interleave two outputs."""
        work = tmp_path / "work"
        builder = HistoricalFeatureMatrixBuilder(store, chunk_size=5)
        builder.build_chunks(hourly(12), work, PROVIDERS, CONFIG)

        different, _, chunks = builder.build_chunks(hourly(6), work, PROVIDERS, CONFIG)
        assert len(different) == len(hourly(6))
        assert len(chunks) == 2

    def test_a_corrupt_progress_file_restarts_rather_than_failing(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / PROGRESS_FILENAME).write_text("{not json", encoding="utf-8")
        rows, _, _ = HistoricalFeatureMatrixBuilder(store, chunk_size=5).build_chunks(
            hourly(8), work, PROVIDERS, CONFIG
        )
        assert len(rows) == len(hourly(8))

    def test_resume_disabled_rebuilds_from_scratch(self, store: IntelligenceStore, tmp_path: Path) -> None:
        work = tmp_path / "work"
        builder = HistoricalFeatureMatrixBuilder(store, chunk_size=5)
        first, _, _ = builder.build_chunks(hourly(12), work, PROVIDERS, CONFIG)
        second, _, _ = builder.build_chunks(hourly(12), work, PROVIDERS, CONFIG, resume=False)
        assert first == second


class TestDatasetCatalog:
    def test_an_entry_round_trips(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service = HistoricalDatasetService(store, chunk_size=6)
        result = service.build(
            hourly(12), tmp_path / "m.parquet", tmp_path / "m.json", PROVIDERS, CONFIG, git_sha="abc"
        )
        fetched = service.catalog.get(result.catalog_entry.dataset_id)
        assert fetched is not None
        assert fetched.row_count == result.manifest.row_count
        assert fetched.git_sha == "abc"
        assert fetched.origin_frequency == "HOURLY"

    def test_re_registering_identical_contents_is_allowed(self, store: IntelligenceStore) -> None:
        """A reproduced build should not be punished."""
        service = HistoricalDatasetService(store)
        entry = _sample_entry()
        service.catalog.register(entry)
        assert service.catalog.register(entry).dataset_id == entry.dataset_id

    def test_registering_a_different_dataset_under_one_id_is_refused(
        self, store: IntelligenceStore
    ) -> None:
        service = HistoricalDatasetService(store)
        entry = _sample_entry()
        service.catalog.register(entry)
        conflicting = entry.model_copy(update={"row_count": 999})
        with pytest.raises(ReplayIntegrityError, match="immutable"):
            service.catalog.register(conflicting)

    def test_catalog_records_every_required_field(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service = HistoricalDatasetService(store, chunk_size=6)
        entry = service.build(
            hourly(8), tmp_path / "m.parquet", tmp_path / "m.json", PROVIDERS, CONFIG, git_sha="abc"
        ).catalog_entry
        for field in (
            "dataset_id",
            "origin_start",
            "origin_end",
            "origin_frequency",
            "row_count",
            "feature_contract_version",
            "configuration_fingerprint",
            "source_fingerprint",
            "membership_hash",
            "extractor_versions",
            "output_format",
            "output_file_hash",
            "created_at",
            "git_sha",
        ):
            assert getattr(entry, field) is not None, field


class TestManifestIntegrity:
    def test_the_manifest_is_written_last(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service = HistoricalDatasetService(store, chunk_size=6)
        result = service.build(
            hourly(12), tmp_path / "m.parquet", tmp_path / "m.json", PROVIDERS, CONFIG, git_sha="abc"
        )
        assert result.manifest_path.stat().st_mtime_ns >= result.output_path.stat().st_mtime_ns

    def test_the_manifest_hash_matches_the_output(self, store: IntelligenceStore, tmp_path: Path) -> None:
        import hashlib

        service = HistoricalDatasetService(store, chunk_size=6)
        result = service.build(
            hourly(12), tmp_path / "m.parquet", tmp_path / "m.json", PROVIDERS, CONFIG, git_sha="abc"
        )
        on_disk = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert on_disk["file_hash"] == hashlib.sha256(result.output_path.read_bytes()).hexdigest()

    def test_the_manifest_records_the_implementation_mode(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        service = HistoricalDatasetService(store, chunk_size=6)
        optimized = service.build(
            hourly(8), tmp_path / "o.parquet", tmp_path / "o.json", PROVIDERS, CONFIG, git_sha="x"
        )
        reference = service.build(
            hourly(8),
            tmp_path / "r.parquet",
            tmp_path / "r.json",
            PROVIDERS,
            CONFIG,
            git_sha="x",
            mode=ReplayMode.REFERENCE,
        )
        assert optimized.manifest.mode == "OPTIMIZED"
        assert reference.manifest.mode == "REFERENCE"

    def test_both_modes_yield_the_same_dataset_id(self, store: IntelligenceStore, tmp_path: Path) -> None:
        """Mode is provenance, not semantics. Identical inputs, identical id."""
        service = HistoricalDatasetService(store, chunk_size=6)
        optimized = service.build(
            hourly(8), tmp_path / "o.parquet", tmp_path / "o.json", PROVIDERS, CONFIG, git_sha="x"
        )
        reference = service.build(
            hourly(8),
            tmp_path / "r.parquet",
            tmp_path / "r.json",
            PROVIDERS,
            CONFIG,
            git_sha="x",
            mode=ReplayMode.REFERENCE,
        )
        assert optimized.manifest.dataset_id == reference.manifest.dataset_id

    def test_the_scratch_directory_is_removed_on_success(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        service = HistoricalDatasetService(store, chunk_size=4)
        result = service.build(
            hourly(12), tmp_path / "m.parquet", tmp_path / "m.json", PROVIDERS, CONFIG, git_sha="x"
        )
        assert not (result.output_path.parent / f".{result.output_path.name}.parts").exists()

    def test_csv_export_is_supported(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service = HistoricalDatasetService(store, chunk_size=6)
        result = service.build(
            hourly(6),
            tmp_path / "m.csv",
            tmp_path / "m.json",
            PROVIDERS,
            CONFIG,
            export_format="csv",
            git_sha="x",
        )
        header = result.output_path.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(MATRIX_COLUMNS)

    def test_an_unknown_format_is_rejected(self, store: IntelligenceStore, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="parquet or csv"):
            HistoricalDatasetService(store).build(
                hourly(4), tmp_path / "m.x", tmp_path / "m.json", PROVIDERS, CONFIG, export_format="avro"
            )


class TestIncrementalExtension:
    def _existing(self, store: IntelligenceStore, tmp_path: Path):
        service = HistoricalDatasetService(store, chunk_size=6)
        result = service.build(
            hourly(24), tmp_path / "m.parquet", tmp_path / "m.json", PROVIDERS, CONFIG, git_sha="x"
        )
        return service, result

    def test_only_new_origins_are_appended(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service, existing = self._existing(store, tmp_path)
        extended = service.extend(
            existing.catalog_entry.dataset_id,
            hourly(24, hours_forward=6),
            tmp_path / "m2.parquet",
            tmp_path / "m2.json",
            PROVIDERS,
            CONFIG,
            git_sha="x",
        )
        assert extended.rows_appended == 6
        assert extended.manifest.row_count == existing.manifest.row_count + 6

    def test_the_original_dataset_is_not_overwritten(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service, existing = self._existing(store, tmp_path)
        service.extend(
            existing.catalog_entry.dataset_id,
            hourly(24, hours_forward=6),
            tmp_path / "m2.parquet",
            tmp_path / "m2.json",
            PROVIDERS,
            CONFIG,
            git_sha="x",
        )
        assert service.catalog.get(existing.catalog_entry.dataset_id) is not None
        assert existing.output_path.exists()

    def test_a_changed_configuration_fails_closed(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service, existing = self._existing(store, tmp_path)
        with pytest.raises(ReplayIntegrityError, match="configuration fingerprint changed"):
            service.extend(
                existing.catalog_entry.dataset_id,
                hourly(24, hours_forward=6),
                tmp_path / "m2.parquet",
                tmp_path / "m2.json",
                PROVIDERS,
                "a-different-configuration",
                git_sha="x",
            )

    def test_a_changed_cadence_fails_closed(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service, existing = self._existing(store, tmp_path)
        four_hourly = generate_origins(
            BASE - timedelta(hours=24), BASE + timedelta(hours=8), OriginFrequency.FOUR_HOURLY
        )
        with pytest.raises(ReplayIntegrityError, match="cadence"):
            service.extend(
                existing.catalog_entry.dataset_id,
                four_hourly,
                tmp_path / "m2.parquet",
                tmp_path / "m2.json",
                PROVIDERS,
                CONFIG,
                git_sha="x",
            )

    def test_a_changed_source_history_fails_closed(self, store: IntelligenceStore, tmp_path: Path) -> None:
        """Backfilled evidence inside the covered window would change rows that
        are already published, so the extension must refuse."""
        from tests.test_intelligence_replay_engine import make_document, make_event

        service, existing = self._existing(store, tmp_path)
        backfill_moment = BASE - timedelta(hours=20)
        document = make_document(9001, backfill_moment, document_id="doc-backfilled")
        store.put_documents([document])
        store.put_signals(
            [make_event(9001, backfill_moment, source_ids=("doc-backfilled",), event_id="event-backfilled")]
        )

        with pytest.raises(ReplayIntegrityError, match="source history changed"):
            service.extend(
                existing.catalog_entry.dataset_id,
                hourly(24, hours_forward=6),
                tmp_path / "m2.parquet",
                tmp_path / "m2.json",
                PROVIDERS,
                CONFIG,
                git_sha="x",
            )

    def test_a_gap_in_the_cadence_fails_closed(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service, existing = self._existing(store, tmp_path)
        gapped = generate_origins(
            BASE + timedelta(hours=3), BASE + timedelta(hours=8), OriginFrequency.HOURLY
        )
        with pytest.raises(ReplayIntegrityError, match="continue the cadence"):
            service.extend(
                existing.catalog_entry.dataset_id,
                gapped,
                tmp_path / "m2.parquet",
                tmp_path / "m2.json",
                PROVIDERS,
                CONFIG,
                git_sha="x",
            )

    def test_extending_with_nothing_new_is_refused(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service, existing = self._existing(store, tmp_path)
        with pytest.raises(ReplayIntegrityError, match="nothing to extend"):
            service.extend(
                existing.catalog_entry.dataset_id,
                hourly(24),
                tmp_path / "m2.parquet",
                tmp_path / "m2.json",
                PROVIDERS,
                CONFIG,
                git_sha="x",
            )

    def test_an_unknown_dataset_is_refused(self, store: IntelligenceStore, tmp_path: Path) -> None:
        service = HistoricalDatasetService(store)
        with pytest.raises(ReplayIntegrityError, match="unknown dataset"):
            service.extend(
                "0" * 64,
                hourly(6),
                tmp_path / "m.parquet",
                tmp_path / "m.json",
                PROVIDERS,
                CONFIG,
            )


class TestFingerprints:
    def test_source_fingerprint_changes_when_evidence_is_added(self, store: IntelligenceStore) -> None:
        from tests.test_intelligence_replay_engine import make_document

        before = source_fingerprint(store, BASE)
        store.put_documents([make_document(7001, BASE - timedelta(hours=2), document_id="doc-new")])
        assert source_fingerprint(store, BASE) != before

    def test_source_fingerprint_is_bounded_by_the_horizon(self, store: IntelligenceStore) -> None:
        """Evidence after the horizon must not change a fingerprint taken before it."""
        from tests.test_intelligence_replay_engine import make_document

        horizon = BASE - timedelta(hours=10)
        before = source_fingerprint(store, horizon)
        store.put_documents([make_document(7002, BASE + timedelta(days=5), document_id="doc-future")])
        assert source_fingerprint(store, horizon) == before

    def test_membership_hash_is_order_sensitive_and_deterministic(self) -> None:
        assert membership_hash(["a", "b"]) == membership_hash(["a", "b"])
        assert membership_hash(["a", "b"]) != membership_hash(["b", "a"])


def _sample_entry():
    from market_intelligence.catalog import DatasetCatalogEntry

    return DatasetCatalogEntry(
        dataset_id="d" * 64,
        origin_start=BASE,
        origin_end=BASE + timedelta(hours=5),
        origin_frequency="HOURLY",
        row_count=6,
        feature_contract_version="market-intelligence-features-v1",
        configuration_fingerprint=CONFIG,
        source_fingerprint="s" * 64,
        membership_hash="m" * 64,
        extractor_versions=("fixture-v1",),
        output_format="parquet",
        output_file_hash="f" * 64,
        created_at=BASE,
        git_sha="abc",
    )


class TestQueryComplexityRegression:
    """B3.1.20. A contract, not an exact count: implementation details may
    legitimately change, but bulk replay must not return to one history scan
    per origin."""

    def test_sql_statements_do_not_grow_with_origin_count(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        counts = {}
        for origin_count in (6, 24):
            counter = _CountingConnection(store.connection)
            store.connection = counter  # type: ignore[assignment]
            try:
                HistoricalFeatureMatrixBuilder(store, chunk_size=100).build_chunks(
                    hourly(origin_count - 1), tmp_path / f"w{origin_count}", PROVIDERS, CONFIG
                )
            finally:
                store.connection = counter.inner  # type: ignore[assignment]
            counts[origin_count] = counter.execute_count

        # Quadrupling the origins must not multiply the statement count. A small
        # additive difference is fine (chunking, dedupe checks); proportional
        # growth means the per-origin query pattern came back.
        assert counts[24] <= counts[6] + 4, counts


class _CountingConnection:
    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.execute_count = 0

    def execute(self, sql: str, *args: object, **kwargs: object) -> object:
        self.execute_count += 1
        return self.inner.execute(sql, *args, **kwargs)  # type: ignore[attr-defined]

    def executemany(self, sql: str, *args: object, **kwargs: object) -> object:
        self.execute_count += 1
        return self.inner.executemany(sql, *args, **kwargs)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


class TestApiAndCliShareOneService:
    """B3.1.22: the API and CLI must not have separate implementations."""

    def test_the_api_builds_the_same_dataset_the_cli_does(
        self, store: IntelligenceStore, tmp_path: Path
    ) -> None:
        from fastapi.testclient import TestClient

        from market_intelligence.api import create_app

        database = store.path
        store.close()
        client = TestClient(create_app(database))
        response = client.post(
            "/replay/dataset",
            json={
                "start": (BASE - timedelta(hours=6)).isoformat(),
                "end": BASE.isoformat(),
                "frequency": "HOURLY",
                "output_path": str(tmp_path / "api.parquet"),
                "manifest_path": str(tmp_path / "api.json"),
                "configuration_fingerprint": CONFIG,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["row_count"] == 7
        assert payload["mode"] == "OPTIMIZED"

        listed = client.get("/datasets").json()
        assert any(entry["dataset_id"] == payload["dataset_id"] for entry in listed)

        fetched = client.get(f"/datasets/{payload['dataset_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["row_count"] == 7

    def test_an_unknown_dataset_is_a_404(self, store: IntelligenceStore) -> None:
        from fastapi.testclient import TestClient

        from market_intelligence.api import create_app

        database = store.path
        store.close()
        client = TestClient(create_app(database))
        assert client.get(f"/datasets/{'0' * 64}").status_code == 404
