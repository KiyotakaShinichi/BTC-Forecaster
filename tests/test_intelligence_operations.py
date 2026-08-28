import io
import json
import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_intelligence.backfill import BackfillRunner
from market_intelligence.cli import build_parser, main
from market_intelligence.configuration import (
    EntityType,
    ProviderCategory,
    ProviderConfig,
    ProviderRegistry,
    QueryPlanner,
    QuerySpec,
    WatchEntity,
)
from market_intelligence.cycle import ReplayService, StructuredRunLogger, run_intelligence_cycle
from market_intelligence.evaluation import GoldLabel, compare_extractors, evaluate_extractor
from market_intelligence.extractors import ExtractionError, FixtureExtractor, RuleBasedExtractor, StructuredLlmExtractor
from market_intelligence.features import FEATURE_CONTRACT_VERSION, FEATURE_DEFINITIONS
from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    ExtractionMethod,
    SignalCategory,
    SourceMetadata,
    SourceType,
    TransferContext,
)
from market_intelligence.operations import (
    BackfillManifest,
    HealthState,
    IntelligenceSnapshot,
    QuarantineRecord,
    RunStatus,
    Watermark,
    WatermarkStatus,
)
from market_intelligence.providers import FixtureSearchProvider, SocialStatementProvider, WhaleDataProvider
from market_intelligence.quality import evaluate_quality
from market_intelligence.retrieval import MultiProviderRetriever, canonical_url, deduplicate_across_providers
from market_intelligence.storage import IntelligenceStore

UTC = timezone.utc
NOW = datetime(2026, 8, 28, 8, tzinfo=UTC)


def doc(identifier="d1", provider="p1", query="btc", available=None, **changes):
    available = available or NOW - timedelta(hours=1)
    data = dict(
        document_id=identifier,
        url=f"https://example.com/{identifier}",
        publisher="Example",
        title="SEC regulation applies to Bitcoin",
        published_at=available - timedelta(minutes=5),
        retrieved_at=available,
        available_at=available,
        text_hash=Document.content_hash(identifier),
        query=query,
        provider=provider,
    )
    data.update(changes)
    return Document(**data)


def event(identifier="e1", source_ids=("d1",), available=None, **changes):
    available = available or NOW - timedelta(minutes=30)
    data = dict(
        event_id=identifier,
        event_time=available,
        available_time=available,
        source_ids=source_ids,
        category=SignalCategory.WEB_EVENT,
        event_type=EventType.REGULATION,
        direction=Direction.UNKNOWN,
        sentiment=0.0,
        btc_relevance=0.6,
        novelty=0.5,
        confidence=0.5,
        expected_horizon_hours=24,
        summary="event",
        extractor_version="fixture-v1",
        extraction_method=ExtractionMethod.FIXTURE,
    )
    data.update(changes)
    return EventSignal(**data)


def query(identifier="q1", text="btc"):
    return QuerySpec(
        query_id=identifier,
        query=text,
        topic="Bitcoin",
        entities=("Bitcoin",),
        event_types=(EventType.REGULATION,),
        lookback_hours=24,
        priority=50,
        generated_at=NOW,
    )


def config(identifier="p1", provider_type="fixture", timeout=1):
    return ProviderConfig(
        id=identifier, type=provider_type, priority=1, timeout=timeout, source_category=ProviderCategory.NEWS
    )


class ConfigurationTests(unittest.TestCase):
    def test_registry_builds_enabled_and_rejects_unknown_type(self):
        registry = ProviderRegistry({"fixture": lambda c: FixtureSearchProvider([])})
        self.assertEqual(list(registry.build([config()])), ["p1"])
        with self.assertRaisesRegex(ValueError, "unknown provider type"):
            registry.build([config(provider_type="mystery")])

    def test_registry_requires_environment_credential(self):
        registry = ProviderRegistry({"fixture": lambda c: object()})
        with self.assertRaisesRegex(ValueError, "not set"):
            registry.build([config().model_copy(update={"credentials_env": "BTC_TEST_MISSING_SECRET"})])

    def test_watchlist_and_query_planner_are_deterministic(self):
        watch = [
            WatchEntity(
                canonical_name="Jerome Powell",
                aliases=("Powell",),
                entity_type=EntityType.CENTRAL_BANK,
                topics=("Federal Reserve", "interest rates"),
                expected_event_types=(EventType.MONETARY_POLICY,),
                importance_prior=0.7,
                measured_impact=None,
            )
        ]
        first, second = QueryPlanner().plan(watch, NOW), QueryPlanner().plan(watch, NOW)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertIsNone(watch[0].measured_impact)


class RetrievalTests(unittest.TestCase):
    def test_retry_backoff_hook_and_eventual_success(self):
        class Flaky(FixtureSearchProvider):
            def __init__(self):
                super().__init__([])
                self.calls = 0

            def search(self, *args):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("temporary")
                return [doc(query="btc")]

        provider, sleeps = Flaky(), []
        result = MultiProviderRetriever(
            {"p1": provider}, {"p1": config()}, max_retries=2, sleep=sleeps.append, jitter=lambda: 0.0
        ).retrieve([query()])
        self.assertTrue(result.attempts[0].success)
        self.assertEqual((provider.calls, sleeps), (3, [1.0, 2.0]))

    def test_provider_failure_is_partial_success(self):
        class Failed(FixtureSearchProvider):
            def search(self, *args):
                raise RuntimeError("down")

        configs = {"a": config("a"), "b": config("b")}
        providers = {"a": FixtureSearchProvider([doc(provider="a")]), "b": Failed([])}
        result = MultiProviderRetriever(providers, configs, max_retries=0, sleep=lambda _: None).retrieve([query()])
        self.assertEqual((result.queries_successful, result.queries_failed, len(result.documents)), (1, 1, 1))

    def test_cross_provider_dedup_preserves_provenance(self):
        a = doc(provider="a", url="https://EXAMPLE.com/story?utm_source=a", text_hash="same")
        b = doc(
            "vendor-id",
            provider="b",
            url="https://example.com/story",
            text_hash="same",
            retrieved_at=NOW - timedelta(minutes=30),
            available_at=NOW - timedelta(minutes=30),
        )
        result = deduplicate_across_providers([a, b])
        self.assertEqual(len(result), 1)
        self.assertEqual({p.provider for p in result[0].retrieval_provenance}, {"a", "b"})
        self.assertEqual(canonical_url(str(a.url)), canonical_url(str(b.url)))

    def test_similar_title_alone_does_not_merge(self):
        a = doc(title="Bitcoin rises today", publisher="A", text_hash="a")
        b = doc("d2", title="Bitcoin rises today", publisher="B", text_hash="b")
        self.assertEqual(len(deduplicate_across_providers([a, b])), 2)

    def test_source_metadata_is_operational_not_political(self):
        metadata = SourceMetadata(
            source_type=SourceType.PRIMARY_OFFICIAL,
            official_source=True,
            primary_source=True,
            known_publisher=True,
            timestamp_quality=1,
            content_completeness=0.8,
        )
        self.assertTrue(doc(source_metadata=metadata).source_metadata.official_source)


class ExtractionTests(unittest.TestCase):
    def test_llm_rejects_nan_and_future_time(self):
        raw_nan = event(extractor_version="v2").model_dump(mode="json")
        raw_nan["sentiment"] = float("nan")
        with self.assertRaises(ExtractionError):
            StructuredLlmExtractor(lambda _: json.dumps([raw_nan]), "v2", now=lambda: NOW).extract([doc()])
        raw_future = event(available=NOW + timedelta(seconds=1), extractor_version="v2").model_dump(mode="json")
        with self.assertRaises(ExtractionError):
            StructuredLlmExtractor(lambda _: json.dumps([raw_future]), "v2", now=lambda: NOW).extract([doc()])

    def test_rule_fallback_records_method_and_unknown_direction(self):
        signals = RuleBasedExtractor({"SEC": ("Securities and Exchange Commission",)}).extract([doc()])
        self.assertEqual(signals[0].extraction_method, ExtractionMethod.RULE_BASED)
        self.assertEqual(signals[0].direction, Direction.UNKNOWN)

    def test_rule_whale_unknown_context(self):
        signal = RuleBasedExtractor().extract([doc(title="Large transfer by Bitcoin whale")])[0]
        self.assertEqual(
            (signal.event_type, signal.transfer_context, signal.direction),
            (EventType.WHALE_TRANSFER, TransferContext.UNKNOWN, Direction.UNKNOWN),
        )

    def test_social_and_whale_are_interfaces(self):
        self.assertTrue(SocialStatementProvider.__abstractmethods__)
        self.assertTrue(WhaleDataProvider.__abstractmethods__)


class PersistenceReplayTests(unittest.TestCase):
    def test_watermark_advances_only_after_atomic_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IntelligenceStore(Path(tmp) / "x.db")
            watermark = Watermark(
                provider_id="p1",
                query_id="q1",
                last_successful_available_time=NOW,
                last_retrieval_time=NOW,
                last_document_id="d1",
                status=WatermarkStatus.SUCCESS,
            )
            original = store.put_signals
            store.put_signals = lambda _: (_ for _ in ()).throw(RuntimeError("disk"))
            with self.assertRaises(RuntimeError):
                store.persist_cycle([doc()], [event()], [watermark])
            self.assertEqual(store.documents_as_of(NOW), [])
            self.assertIsNone(store.get_watermark("p1", "q1"))
            store.put_signals = original
            store.persist_cycle([doc()], [event()], [watermark])
            self.assertEqual(store.get_watermark("p1", "q1").last_document_id, "d1")
            store.close()

    def test_failed_watermark_does_not_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IntelligenceStore(Path(tmp) / "x.db")
            failed = Watermark(provider_id="p1", query_id="q1", last_retrieval_time=NOW, status=WatermarkStatus.FAILED)
            store.persist_cycle([], [], [failed])
            self.assertIsNone(store.get_watermark("p1", "q1"))
            store.close()

    def test_replay_and_snapshot_exclude_future_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IntelligenceStore(Path(tmp) / "x.db")
            past, future = (
                doc(),
                doc("future", available=NOW + timedelta(seconds=1), retrieved_at=NOW + timedelta(seconds=1)),
            )
            past_event, future_event = event(), event("future-event", ("future",), available=NOW + timedelta(seconds=1))
            store.put_documents([past, future])
            store.put_signals([past_event, future_event])
            snap = ReplayService(store).replay(NOW, {"p1": "v1"}, "cfg")
            self.assertEqual((snap.document_ids, snap.event_ids), (("d1",), ("e1",)))
            self.assertEqual(snap.feature_contract_version, FEATURE_CONTRACT_VERSION)
            store.close()

    def test_snapshot_fingerprint_is_deterministic_and_membership_sensitive(self):
        kwargs = dict(
            forecast_origin=NOW,
            document_ids=["d1"],
            event_ids=["e1"],
            features={"x": 1.0},
            provider_versions={"p": "1"},
            extractor_versions=["x"],
            configuration_fingerprint="c",
            source_hashes=["h"],
            feature_contract_version="v1",
        )
        first = IntelligenceSnapshot.create(**kwargs)
        second = IntelligenceSnapshot.create(**kwargs)
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertNotEqual(
            first.snapshot_id, IntelligenceSnapshot.create(**{**kwargs, "event_ids": ["e2"]}).snapshot_id
        )

    def test_backfill_is_bounded_and_resumable(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            initial = BackfillManifest(from_time=NOW, to_time=NOW + timedelta(hours=5), window_hours=2, max_windows=2)
            first = BackfillRunner(lambda a, b: calls.append((a, b))).run(initial, path)
            self.assertEqual(len(calls), 2)
            BackfillRunner(lambda a, b: calls.append((a, b))).run(first.model_copy(update={"max_windows": 5}), path)
            self.assertEqual(len(set(calls)), 3)


class OperationsTests(unittest.TestCase):
    def test_feature_contract_is_explicit_and_unique(self):
        self.assertEqual(len({d.name for d in FEATURE_DEFINITIONS}), 7)
        self.assertTrue(all(d.version == "1.0.0" for d in FEATURE_DEFINITIONS))

    def test_quality_reports_duplicates_orphans_future_and_versions(self):
        bad_event = event(source_ids=("missing",), schema_version="old", available=NOW + timedelta(seconds=1))
        report = evaluate_quality([doc(), doc()], [bad_event], NOW)
        codes = {issue.code for issue in report.issues}
        self.assertTrue({"DUPLICATE_DOCUMENT_ID", "ORPHANED_EVENT", "FUTURE_AVAILABILITY", "SCHEMA_VERSION"} <= codes)

    def test_quarantine_hashes_raw_without_storing_it(self):
        record = QuarantineRecord.from_raw("bad", "p", NOW, "secret raw response", "MALFORMED")
        self.assertNotIn("secret", record.model_dump_json())
        with tempfile.TemporaryDirectory() as tmp:
            store = IntelligenceStore(Path(tmp) / "x.db")
            store.put_quarantine([record])
            self.assertEqual(store.quarantine_count(), 1)
            store.close()

    def test_structured_logger_redacts_secret_fields(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.Logger("test")
        logger.addHandler(handler)
        StructuredRunLogger(logger).emit("attempt", provider="p", api_key="secret", authorization="token")
        self.assertIn('"provider": "p"', stream.getvalue())
        self.assertNotIn("secret", stream.getvalue())

    def test_cycle_partial_failure_missingness_health_and_manifest(self):
        class Down(FixtureSearchProvider):
            def search(self, *args):
                raise RuntimeError("offline")

        providers = {"p1": FixtureSearchProvider([doc()]), "p2": Down([])}
        configs = {"p1": config("p1"), "p2": config("p2")}
        with tempfile.TemporaryDirectory() as tmp:
            store = IntelligenceStore(Path(tmp) / "x.db")
            manifest = Path(tmp) / "manifest.json"
            report = run_intelligence_cycle(
                [query()],
                MultiProviderRetriever(providers, configs, max_retries=0),
                RuleBasedExtractor(),
                store,
                configs,
                NOW,
                manifest,
                now=lambda: NOW,
                source_sha="abc",
            )
            self.assertEqual((report.missingness.queries_successful, report.missingness.queries_failed), (1, 1))
            self.assertNotEqual(report.missingness.coverage_ratio, 0)
            self.assertEqual({h.state for h in report.provider_health}, {HealthState.HEALTHY, HealthState.UNAVAILABLE})
            self.assertEqual(report.manifest.status, RunStatus.PARTIAL_SUCCESS)
            self.assertTrue(manifest.exists())
            self.assertEqual(json.loads(manifest.read_text())["software_source_sha"], "abc")
            store.close()

    def test_extractor_failure_keeps_document_and_quarantines(self):
        class BadExtractor(FixtureExtractor):
            version = "bad"

            def extract(self, documents):
                raise ExtractionError("bad json")

        with tempfile.TemporaryDirectory() as tmp:
            store = IntelligenceStore(Path(tmp) / "x.db")
            report = run_intelligence_cycle(
                [query()],
                MultiProviderRetriever({"p1": FixtureSearchProvider([doc()])}, {"p1": config()}, max_retries=0),
                BadExtractor([]),
                store,
                {},
                NOW,
                now=lambda: NOW,
            )
            self.assertEqual((len(store.documents_as_of(NOW)), len(report.events), store.quarantine_count()), (1, 0, 1))
            self.assertEqual(report.manifest.status, RunStatus.DEGRADED)
            store.close()


class CliAndEvaluationTests(unittest.TestCase):
    def test_cli_parser_has_all_commands(self):
        for command in ("collect", "extract", "aggregate", "replay", "backfill", "health", "quality"):
            with self.subTest(command=command):
                args = [command]
                if command in {"extract", "aggregate", "replay", "quality"}:
                    args += ["--origin", NOW.isoformat()]
                elif command == "collect":
                    args += ["--config", "x", "--origin", NOW.isoformat(), "--manifest", "m"]
                elif command == "backfill":
                    args += ["--from", NOW.isoformat(), "--to", (NOW + timedelta(hours=1)).isoformat(), "--config", "x"]
                self.assertEqual(build_parser().parse_args(args).command, command)

    def test_cli_replay_calls_shared_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--db", str(Path(tmp) / "x.db"), "replay", "--origin", NOW.isoformat()]), 0)

    def test_gold_evaluation_and_comparison(self):
        documents = [doc()]
        labels = [
            GoldLabel(
                case_id="reg",
                document_ids=("d1",),
                expected_event_type=EventType.REGULATION,
                expected_entity="SEC",
                relevance_min=0.5,
                relevance_max=0.7,
            )
        ]
        extractor = RuleBasedExtractor({"SEC": ()})
        result = evaluate_extractor(extractor, documents, labels)
        self.assertEqual((result.event_type_accuracy, result.source_provenance_accuracy), (1.0, 1.0))
        self.assertEqual(len(compare_extractors([extractor], documents, labels).results), 1)


if __name__ == "__main__":
    unittest.main()
