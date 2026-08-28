from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Direction, Document, EventSignal, EventType, ExtractionMethod, SignalCategory
from .replay_dataset import ReplayDatasetBuilder
from .services import SnapshotService
from .storage import IntelligenceStore


def run_local_benchmark(record_count: int = 1000, origin_count: int = 1000) -> dict[str, float | int]:
    """Representative local measurement, not a CI performance assertion."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    documents = []
    events = []
    for index in range(record_count):
        available = base - timedelta(minutes=index % 120)
        document_id = f"benchmark-document-{index}"
        text_hash = Document.content_hash(document_id)
        documents.append(
            Document(
                document_id=document_id,
                url=f"https://example.invalid/benchmark/{index}",
                publisher="Synthetic Benchmark",
                title=f"Synthetic benchmark item {index}",
                published_at=available,
                retrieved_at=available,
                available_at=available,
                text_hash=text_hash,
                query="synthetic benchmark",
                provider="benchmark",
            )
        )
        events.append(
            EventSignal(
                event_id=f"benchmark-event-{index}",
                event_time=available,
                available_time=available,
                source_ids=(document_id,),
                category=SignalCategory.WEB_EVENT,
                event_type=EventType.REGULATION,
                direction=Direction.UNKNOWN,
                sentiment=0.0,
                btc_relevance=0.5,
                novelty=0.5,
                confidence=0.5,
                expected_horizon_hours=24,
                summary="Synthetic benchmark event",
                extractor_version="benchmark-v1",
                extraction_method=ExtractionMethod.FIXTURE,
            )
        )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        store = IntelligenceStore(directory / "benchmark.duckdb")
        store.put_documents(documents)
        store.put_signals(events)
        started = time.perf_counter()
        store.query_documents(limit=record_count)
        document_query_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        store.query_events(limit=record_count)
        event_query_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        SnapshotService(store).build_snapshot(base, {"benchmark": "v1"}, "benchmark")
        snapshot_ms = (time.perf_counter() - started) * 1000
        origins = [base + timedelta(minutes=index) for index in range(origin_count)]
        started = time.perf_counter()
        ReplayDatasetBuilder(store).build(
            origins,
            directory / "benchmark.parquet",
            directory / "benchmark.manifest.json",
            {"benchmark": "v1"},
            "benchmark",
            git_sha="benchmark",
        )
        replay_ms = (time.perf_counter() - started) * 1000
        store.close()
    return {
        "documents": record_count,
        "events": record_count,
        "origins": origin_count,
        "document_query_ms": round(document_query_ms, 3),
        "event_query_ms": round(event_query_ms, 3),
        "snapshot_ms": round(snapshot_ms, 3),
        "replay_dataset_ms": round(replay_ms, 3),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_local_benchmark(), indent=2))
