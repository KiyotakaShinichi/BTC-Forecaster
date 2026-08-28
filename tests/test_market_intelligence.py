import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from market_intelligence.aggregation import FeatureAggregator
from market_intelligence.cache import DeterministicCache
from market_intelligence.extractors import ExtractionError, FixtureExtractor, StructuredLlmExtractor
from market_intelligence.models import (
    Direction,
    Document,
    EventSignal,
    EventType,
    SignalCategory,
    TransferContext,
)
from market_intelligence.pipeline import IntelligencePipeline
from market_intelligence.providers import FixtureSearchProvider, deduplicate_documents
from market_intelligence.storage import IntelligenceStore

UTC = timezone.utc
ORIGIN = datetime(2026, 1, 10, 12, tzinfo=UTC)


def document(**updates):
    values = dict(
        document_id="doc-1",
        url="https://example.com/a",
        publisher="Example",
        title="BTC item",
        published_at=ORIGIN - timedelta(hours=3),
        retrieved_at=ORIGIN - timedelta(hours=2),
        available_at=ORIGIN - timedelta(hours=3),
        author="Reporter",
        text_hash=Document.content_hash("body"),
        query="bitcoin",
        provider="fixture",
    )
    values.update(updates)
    return Document(**values)


def signal(**updates):
    values = dict(
        event_id="event-1",
        event_time=ORIGIN - timedelta(hours=4),
        available_time=ORIGIN - timedelta(hours=3),
        source_ids=("doc-1",),
        category=SignalCategory.WEB_EVENT,
        entity=None,
        event_type=EventType.REGULATION,
        asset="BTC",
        direction=Direction.NEUTRAL,
        sentiment=0.5,
        btc_relevance=0.8,
        novelty=0.75,
        confidence=0.9,
        expected_horizon_hours=72,
        summary="A validated event",
        extractor_version="fixture-v1",
    )
    values.update(updates)
    return EventSignal(**values)


class SchemaTests(unittest.TestCase):
    def test_bounded_scores_and_timezone_are_validated(self):
        with self.assertRaises(ValidationError):
            signal(sentiment=1.1)
        with self.assertRaises(ValidationError):
            document(retrieved_at=datetime(2026, 1, 1))

    def test_sentiment_and_relevance_are_separate(self):
        unrelated = signal(sentiment=1.0, btc_relevance=0.0)
        features = FeatureAggregator().aggregate([unrelated], ORIGIN)
        self.assertEqual(features["sentiment_mean_24h"], 1.0)
        self.assertEqual(features["sentiment_weighted_24h"], 0.0)

    def test_unknown_whale_context_cannot_claim_direction(self):
        with self.assertRaises(ValidationError):
            signal(
                event_type=EventType.WHALE_TRANSFER,
                transfer_context=TransferContext.UNKNOWN,
                direction=Direction.BULLISH,
            )
        unknown = signal(
            event_type=EventType.WHALE_TRANSFER, transfer_context=TransferContext.UNKNOWN, direction=Direction.UNKNOWN
        )
        self.assertEqual(unknown.direction, Direction.UNKNOWN)


class ProviderAndTimeTests(unittest.TestCase):
    def test_deduplication_uses_stable_content_fingerprint(self):
        later = document(document_id="vendor-other-id", retrieved_at=ORIGIN)
        self.assertEqual(len(deduplicate_documents([document(), later])), 1)
        self.assertEqual(deduplicate_documents([later, document()])[0].document_id, "doc-1")

    def test_fixture_provider_filters_time(self):
        future = document(
            document_id="future",
            url="https://example.com/future",
            available_at=ORIGIN + timedelta(minutes=1),
            retrieved_at=ORIGIN + timedelta(minutes=2),
        )
        found = FixtureSearchProvider([document(), future]).search("bitcoin", ORIGIN - timedelta(days=1), ORIGIN)
        self.assertEqual([d.document_id for d in found], ["doc-1"])

    def test_aggregation_excludes_future_signal(self):
        future = signal(event_id="future", available_time=ORIGIN + timedelta(seconds=1), sentiment=-1)
        result = FeatureAggregator().aggregate([signal(), future], ORIGIN)
        self.assertEqual(result["event_count_24h"], 1.0)
        self.assertEqual(result["sentiment_mean_24h"], 0.5)

    def test_aggregation_windows_and_taxonomies(self):
        macro = signal(
            event_id="macro",
            event_type=EventType.MACRO_SHOCK,
            sentiment=-0.5,
            available_time=ORIGIN - timedelta(hours=48),
        )
        result = FeatureAggregator().aggregate([signal(), macro], ORIGIN)
        self.assertEqual(result["event_count_24h"], 1.0)
        self.assertAlmostEqual(result["regulatory_signal_72h"], 0.5)
        self.assertAlmostEqual(result["macro_news_signal_72h"], -0.5)


class ExtractionAndCacheTests(unittest.TestCase):
    def test_structured_extraction_and_provenance(self):
        raw = json.dumps([signal(extractor_version="llm-v2").model_dump(mode="json")])
        output = StructuredLlmExtractor(lambda _: raw, "llm-v2").extract([document()])
        self.assertEqual(output[0].source_ids, ("doc-1",))

    def test_bad_llm_json_is_rejected(self):
        with self.assertRaises(ExtractionError):
            StructuredLlmExtractor(lambda _: "not json", "v1").extract([document()])

    def test_fabricated_source_id_is_rejected(self):
        raw = json.dumps([signal(source_ids=("fabricated",), extractor_version="llm-v2").model_dump(mode="json")])
        with self.assertRaises(ExtractionError):
            StructuredLlmExtractor(lambda _: raw, "llm-v2").extract([document()])

    def test_signal_cannot_be_available_before_source(self):
        premature = signal(available_time=ORIGIN - timedelta(hours=4), extractor_version="llm-v2")
        raw = json.dumps([premature.model_dump(mode="json")])
        with self.assertRaises(ExtractionError):
            StructuredLlmExtractor(lambda _: raw, "llm-v2").extract([document()])

    def test_cache_key_is_stable_and_versioned(self):
        args = ("bitcoin", ORIGIN - timedelta(days=1), ORIGIN, "fixture", "v1")
        self.assertEqual(DeterministicCache.key(*args), DeterministicCache.key(*args))
        self.assertNotEqual(DeterministicCache.key(*args), DeterministicCache.key(*args[:-1], "v2"))

    def test_pipeline_reuses_cache_without_provider_call(self):
        class CountingProvider(FixtureSearchProvider):
            def __init__(self):
                super().__init__([document()])
                self.calls = 0

            def search(self, *args):
                self.calls += 1
                return super().search(*args)

        provider = CountingProvider()
        with tempfile.TemporaryDirectory() as directory:
            pipeline = IntelligencePipeline(provider, FixtureExtractor([signal()]), DeterministicCache(directory))
            for _ in range(2):
                documents, signals = pipeline.run("bitcoin", ORIGIN - timedelta(days=1), ORIGIN)
            self.assertEqual(provider.calls, 1)
            self.assertEqual((documents[0].document_id, signals[0].event_id), ("doc-1", "event-1"))


class StorageTests(unittest.TestCase):
    def test_duckdb_as_of_filter_and_parquet_export(self):
        future_document = document(
            document_id="future",
            url="https://example.com/future",
            available_at=ORIGIN + timedelta(minutes=1),
            retrieved_at=ORIGIN + timedelta(minutes=2),
        )
        future_signal = signal(event_id="future", available_time=ORIGIN + timedelta(minutes=1))
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "intelligence.duckdb")
            store.put_documents([document(), future_document])
            store.put_signals([signal(), future_signal])
            self.assertEqual([d.document_id for d in store.documents_as_of(ORIGIN)], ["doc-1"])
            self.assertEqual([s.event_id for s in store.signals_as_of(ORIGIN)], ["event-1"])
            parquet = Path(directory) / "parquet"
            store.export_parquet(parquet)
            self.assertTrue((parquet / "documents.parquet").exists())
            self.assertTrue((parquet / "signals.parquet").exists())
            store.close()


if __name__ == "__main__":
    unittest.main()
