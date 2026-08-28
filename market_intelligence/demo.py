from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .configuration import ProviderCategory, ProviderConfig, QuerySpec
from .cycle import run_intelligence_cycle
from .extractors import RuleBasedExtractor
from .models import Document, EventType, SourceMetadata, SourceType
from .providers import FixtureSearchProvider
from .replay_dataset import ReplayDatasetBuilder
from .retrieval import MultiProviderRetriever
from .storage import IntelligenceStore


def run_offline_demo(output_directory: str | Path) -> dict[str, object]:
    """Deterministic synthetic research demo; never represents live intelligence."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    origin = datetime(2025, 1, 15, 12, tzinfo=timezone.utc)
    query = QuerySpec(
        query_id="synthetic-demo-query",
        query="synthetic bitcoin regulation",
        topic="Bitcoin regulation",
        entities=("SEC",),
        event_types=(EventType.REGULATION,),
        lookback_hours=24,
        priority=1,
        generated_at=origin,
    )
    text_hash = Document.content_hash("CURATED SYNTHETIC FIXTURE: no live claim")
    document = Document(
        document_id=Document.stable_id("https://example.invalid/synthetic-demo", text_hash),
        url="https://example.invalid/synthetic-demo",
        publisher="Synthetic Fixture Publisher",
        title="SEC regulation applies to Bitcoin — synthetic demo",
        published_at=origin - timedelta(hours=2),
        retrieved_at=origin - timedelta(hours=1),
        available_at=origin - timedelta(hours=1),
        author="Fixture Generator",
        text_hash=text_hash,
        query=query.query,
        provider="synthetic-demo",
        source_metadata=SourceMetadata(source_type=SourceType.UNKNOWN, content_completeness=1.0),
    )
    provider = FixtureSearchProvider([document])
    provider.name = "synthetic-demo"
    config = ProviderConfig(
        id="synthetic-demo", type="fixture", priority=1, timeout=5, source_category=ProviderCategory.NEWS
    )
    database = output / "demo-intelligence.duckdb"
    store = IntelligenceStore(database)
    try:
        report = run_intelligence_cycle(
            [query],
            MultiProviderRetriever({"synthetic-demo": provider}, {"synthetic-demo": config}, max_retries=0),
            RuleBasedExtractor({"SEC": ()}),
            store,
            {"namespace": "SYNTHETIC_DEMO_V1"},
            origin,
            output / "run-manifest.json",
            source_sha="offline-demo",
        )
        origins = [origin, origin + timedelta(hours=6)]
        dataset = ReplayDatasetBuilder(store).build(
            origins,
            output / "replay-features.parquet",
            output / "replay-features.manifest.json",
            {"synthetic-demo": "fixture-v1"},
            "SYNTHETIC_DEMO_V1",
            git_sha="offline-demo",
        )
        return {
            "label": "MARKET INTELLIGENCE / RESEARCH — SYNTHETIC FIXTURES — NOT INVESTMENT ADVICE",
            "run_id": report.run_id,
            "dataset_id": dataset.dataset_id,
            "database": str(database),
            "dataset": str(output / "replay-features.parquet"),
        }
    finally:
        store.close()
