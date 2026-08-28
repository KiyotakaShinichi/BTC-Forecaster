import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
from fastapi.testclient import TestClient

from market_intelligence.api import create_app
from market_intelligence.errors import ReplayIntegrityError, StorageError
from market_intelligence.extractors import RuleBasedExtractor
from market_intelligence.gold import load_gold_set, write_gold_evaluation
from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    ExtractionMethod,
    SignalCategory,
)
from market_intelligence.operations import (
    HealthState,
    ProviderHealth,
    QualityScoreboard,
    QuarantineRecord,
    RunManifest,
    RunStatus,
    Watermark,
    WatermarkStatus,
)
from market_intelligence.replay_dataset import ReplayDatasetBuilder
from market_intelligence.retrieval import ProviderAttempt
from market_intelligence.services import IntelligenceReadService
from market_intelligence.storage import SCHEMA_VERSION, IntelligenceStore

UTC = timezone.utc
ORIGIN = datetime(2026, 8, 28, 8, tzinfo=UTC)


def make_document(identifier: str, available_at: datetime, provider: str = "fixture") -> Document:
    return Document(
        document_id=identifier,
        url=f"https://example.invalid/{identifier}",
        publisher="Synthetic Publisher",
        title=f"Synthetic event {identifier}",
        published_at=available_at - timedelta(minutes=5),
        retrieved_at=available_at,
        available_at=available_at,
        text_hash=Document.content_hash(identifier),
        query="synthetic topic",
        provider=provider,
    )


def make_event(identifier: str, document_id: str, available_at: datetime) -> EventSignal:
    return EventSignal(
        event_id=identifier,
        event_time=available_at,
        available_time=available_at,
        source_ids=(document_id,),
        category=SignalCategory.WEB_EVENT,
        entity="Synthetic Entity",
        event_type=EventType.REGULATION,
        direction=Direction.UNKNOWN,
        sentiment=0.0,
        btc_relevance=0.8,
        novelty=0.5,
        confidence=0.7,
        expected_horizon_hours=24,
        summary="Synthetic curated fixture",
        extractor_version="fixture-v1",
        extraction_method=ExtractionMethod.FIXTURE,
    )


def make_run(run_id: str, finished_at: datetime, coverage: float = 1.0) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        started_at=finished_at - timedelta(minutes=1),
        finished_at=finished_at,
        configuration_fingerprint="cfg",
        providers_attempted=1,
        queries_attempted=1,
        documents_accepted=0,
        documents_rejected=0,
        events_accepted=0,
        events_rejected=0,
        quality_summary={
            "valid": True,
            "provider_coverage_ratio": coverage,
            "queries_failed": 0,
            "source_stale_flag": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        },
        watermark_changes=1,
        software_source_sha="test",
        status=RunStatus.SUCCESS,
        provider_ids=("fixture",),
    )


class StorageV3Tests(unittest.TestCase):
    def test_forward_migration_preserves_b2_data_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.duckdb"
            connection = duckdb.connect(str(path))
            connection.execute(
                "CREATE TABLE documents(document_id VARCHAR PRIMARY KEY, available_at TIMESTAMPTZ, payload JSON)"
            )
            document = make_document("legacy", ORIGIN)
            connection.execute("INSERT INTO documents VALUES (?, ?, ?)", ["legacy", ORIGIN, document.model_dump_json()])
            connection.close()
            store = IntelligenceStore(path)
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            self.assertEqual(store.get_document("legacy").document_id, "legacy")
            store.close()
            reopened = IntelligenceStore(path)
            self.assertEqual(reopened.schema_version(), SCHEMA_VERSION)
            reopened.close()

    def test_orphans_watermark_regression_and_run_rewrite_fail_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = IntelligenceStore(Path(temporary) / "integrity.duckdb")
            with self.assertRaises(ReplayIntegrityError):
                store.put_signals([make_event("orphan", "missing", ORIGIN)])
            newer = Watermark(
                provider_id="p",
                query_id="q",
                last_successful_available_time=ORIGIN,
                last_retrieval_time=ORIGIN,
                status=WatermarkStatus.SUCCESS,
            )
            store.persist_cycle([], [], [newer])
            older = newer.model_copy(
                update={
                    "last_successful_available_time": ORIGIN - timedelta(hours=1),
                    "last_retrieval_time": ORIGIN - timedelta(hours=1),
                }
            )
            with self.assertRaises(StorageError):
                store.persist_cycle([], [], [older])
            run = make_run("immutable", ORIGIN)
            store.put_run(run)
            with self.assertRaises(ReplayIntegrityError):
                store.put_run(run.model_copy(update={"documents_accepted": 99}))
            store.close()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = IntelligenceStore(Path(self.temporary.name) / "api.duckdb")
        past = make_document("past", ORIGIN - timedelta(hours=1))
        future = make_document("future", ORIGIN + timedelta(hours=1))
        self.store.put_documents([past, future])
        self.store.put_signals(
            [
                make_event("past-event", "past", ORIGIN - timedelta(minutes=30)),
                make_event("future-event", "future", ORIGIN + timedelta(hours=1)),
            ]
        )
        self.client = TestClient(create_app(store=self.store))

    def tearDown(self):
        self.client.close()
        self.store.close()
        self.temporary.cleanup()

    def test_health_ready_metrics_and_request_correlation(self):
        response = self.client.get("/health", headers={"X-Request-ID": "test-request-1"})
        self.assertEqual(response.json(), {"status": "alive", "schema_version": None})
        self.assertEqual(response.headers["X-Request-ID"], "test-request-1")
        self.assertEqual(self.client.get("/ready").json()["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.client.get("/metrics").json()["documents_ingested"], 2)

    def test_point_in_time_lists_details_timeline_and_trace_exclude_future(self):
        origin = ORIGIN.isoformat()
        documents = self.client.get("/documents", params={"forecast_origin": origin}).json()["items"]
        events = self.client.get("/events", params={"forecast_origin": origin}).json()["items"]
        timeline = self.client.get("/timeline", params={"forecast_origin": origin}).json()["items"]
        self.assertEqual([item["document_id"] for item in documents], ["past"])
        self.assertEqual([item["event_id"] for item in events], ["past-event"])
        self.assertEqual(len(timeline), 1)
        self.assertEqual(self.client.get("/documents/future", params={"forecast_origin": origin}).status_code, 404)
        trace = self.client.get("/events/past-event", params={"forecast_origin": origin}).json()
        self.assertEqual(trace["sources"][0]["document_id"], "past")

    def test_filter_validation_and_bounded_pagination(self):
        response = self.client.get(
            "/events", params={"event_type": "REGULATION", "minimum_relevance": 0.75, "limit": 1}
        )
        self.assertEqual(len(response.json()["items"]), 1)
        self.assertEqual(self.client.get("/events", params={"limit": 501}).status_code, 422)
        self.assertEqual(
            self.client.get(
                "/events",
                params={"available_from": "2020-01-01T00:00:00Z", "available_to": "2022-01-02T00:00:00Z"},
            ).status_code,
            422,
        )

    def test_replay_snapshot_contract_and_mismatch_error(self):
        response = self.client.post(
            "/replay",
            json={"forecast_origin": ORIGIN.isoformat(), "configuration_fingerprint": "cfg"},
        )
        self.assertEqual(response.status_code, 200)
        snapshot = response.json()
        self.assertEqual(snapshot["document_ids"], ["past"])
        verified = self.client.get(f"/snapshots/{snapshot['snapshot_id']}", params={"configuration_fingerprint": "cfg"})
        self.assertEqual(verified.status_code, 200)
        mismatch = self.client.get(
            f"/snapshots/{snapshot['snapshot_id']}", params={"configuration_fingerprint": "wrong"}
        )
        self.assertEqual((mismatch.status_code, mismatch.json()["error"]), (409, "SnapshotMismatchError"))

    def test_dashboard_is_research_only_and_no_secret_surface(self):
        dashboard = self.client.get("/dashboard").text
        self.assertIn("MARKET INTELLIGENCE / RESEARCH", dashboard)
        self.assertIn("Not investment advice", dashboard)
        openapi = json.dumps(self.client.get("/openapi.json").json()).casefold()
        self.assertNotIn("api_key", openapi)
        self.assertNotIn("authorization", openapi)

    def test_run_quality_quarantine_and_observability_endpoints(self):
        run = make_run("run-1", ORIGIN - timedelta(minutes=1))
        self.store.put_run(run)
        self.store.put_quality_scoreboard(
            QualityScoreboard(run_id=run.run_id, created_at=ORIGIN, provider_coverage_ratio=1.0)
        )
        self.store.put_health([ProviderHealth(provider_id="fixture", state=HealthState.HEALTHY)])
        self.store.put_provider_attempts(
            run.run_id,
            [
                ProviderAttempt(
                    provider_id="fixture", query_id="q", success=True, attempts=1, latency_ms=5, documents_received=2
                )
            ],
            ORIGIN,
        )
        self.store.put_quarantine(
            [QuarantineRecord.from_raw("invalid JSON", "fixture", ORIGIN, "redacted", "BAD_EXTRACTOR_OUTPUT")]
        )
        runs = self.client.get("/runs", params={"status": "SUCCESS", "provider": "fixture"}).json()["items"]
        self.assertEqual(runs[0]["run_id"], "run-1")
        self.assertEqual(self.client.get("/quality/latest").json()["run_id"], "run-1")
        self.assertEqual(self.client.get("/providers/health").json()[0]["state"], "HEALTHY")
        self.assertEqual(self.client.get("/providers/observability").json()[0]["success_rate"], 1.0)
        extraction = self.client.get("/extraction/observability").json()
        self.assertIn("confidence_distribution", extraction)
        quarantine = self.client.get("/quarantine", params={"record_type": "BAD_EXTRACTOR_OUTPUT"}).json()
        self.assertEqual(len(quarantine["items"]), 1)
        self.assertEqual(len(self.client.get("/features").json()), 7)

    def test_naive_forecast_origin_is_rejected(self):
        self.assertEqual(
            self.client.get("/documents", params={"forecast_origin": "2026-01-01T00:00:00"}).status_code, 422
        )


class ReplayDatasetTests(unittest.TestCase):
    def test_parquet_manifest_hash_and_separate_missingness(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            store = IntelligenceStore(directory / "dataset.duckdb")
            store.put_run(make_run("covered", ORIGIN - timedelta(minutes=1), coverage=1.0))
            origins = [ORIGIN, ORIGIN + timedelta(hours=1)]
            manifest = ReplayDatasetBuilder(store).build(
                origins,
                directory / "features.parquet",
                directory / "features.manifest.json",
                {"fixture": "v1"},
                "cfg",
                git_sha="test-sha",
            )
            self.assertEqual((manifest.row_count, manifest.origin_count), (2, 2))
            self.assertEqual(
                manifest.file_hash, hashlib.sha256((directory / "features.parquet").read_bytes()).hexdigest()
            )
            rows = (
                duckdb.connect()
                .execute(
                    "SELECT event_count_24h, provider_coverage_ratio, source_stale_flag FROM read_parquet(?)",
                    [str(directory / "features.parquet")],
                )
                .fetchall()
            )
            self.assertEqual(rows[0], (0.0, 1.0, 0))
            self.assertTrue((directory / "features.manifest.json").exists())
            self.assertEqual(store.metrics()["replay_rows_generated"], 2)
            store.close()

    def test_no_run_means_missing_coverage_not_observed_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            store = IntelligenceStore(directory / "missing.duckdb")
            ReplayDatasetBuilder(store).build(
                [ORIGIN], directory / "features.csv", directory / "manifest.json", {}, "cfg", "csv", "test"
            )
            row = (directory / "features.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
            header = (directory / "features.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
            values = dict(zip(header, row, strict=True))
            self.assertEqual(values["event_count_24h"], "0.0")
            self.assertEqual(values["provider_coverage_ratio"], "0.0")
            self.assertEqual(values["source_stale_flag"], "1")
            store.close()


class SharedServiceTests(unittest.TestCase):
    def test_cli_api_shared_read_service_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = IntelligenceStore(Path(temporary) / "shared.duckdb")
            service = IntelligenceReadService(store)
            api = TestClient(create_app(store=store))
            self.assertEqual(
                service.aggregate(ORIGIN),
                api.post(
                    "/replay", json={"forecast_origin": ORIGIN.isoformat(), "configuration_fingerprint": "cfg"}
                ).json()["features"],
            )
            api.close()
            store.close()


class ReproducibilityTests(unittest.TestCase):
    def test_gold_fixture_manifest_hash_and_version(self):
        root = Path(__file__).parents[1] / "market_intelligence"
        fixture_hash = hashlib.sha256((root / "gold_fixtures.json").read_bytes()).hexdigest()
        manifest = json.loads((root / "gold_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["gold_version"], "gold-v1")
        self.assertEqual(manifest["case_count"], 9)
        self.assertEqual(manifest["case_hash"], fixture_hash)
        documents, labels, typed_manifest = load_gold_set()
        self.assertEqual((len(labels), typed_manifest.case_count), (9, 9))
        self.assertEqual(len(documents), 10)

    def test_formal_gold_report_states_small_sample_limitation(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = write_gold_evaluation(
                RuleBasedExtractor({"SEC": (), "CFTC": ()}), Path(temporary) / "gold-report.json"
            )
            self.assertEqual(report.result.cases, 9)
            self.assertIn("not evidence of production accuracy", report.limitation)

    def test_imports_do_not_create_database_or_contact_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(__file__).parents[1]
            command = [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(repository)!r}); "
                "import market_intelligence; import market_intelligence.api; import market_intelligence.demo",
            ]
            result = subprocess.run(command, cwd=temporary, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
