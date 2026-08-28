from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from .aggregation import FeatureAggregator
from .cache import DeterministicCache
from .configuration import QuerySpec, configuration_fingerprint
from .extractors import EventExtractor
from .features import FEATURE_CONTRACT_VERSION
from .models import Document, EventSignal
from .operations import (
    HealthState,
    IntelligenceSnapshot,
    MissingnessReport,
    ProviderHealth,
    QualityScoreboard,
    QuarantineRecord,
    RunManifest,
    RunStatus,
    Watermark,
    WatermarkStatus,
)
from .quality import evaluate_quality
from .retrieval import MultiProviderRetriever, ProviderAttempt, deduplicate_across_providers
from .storage import IntelligenceStore


class IntelligenceRunReport(BaseModel):
    run_id: str
    documents: list[Document]
    events: list[EventSignal]
    features: dict[str, float]
    attempts: list[ProviderAttempt]
    missingness: MissingnessReport
    provider_health: list[ProviderHealth]
    manifest: RunManifest


class StructuredRunLogger:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("btc_intelligence")

    def emit(self, event: str, **fields: object) -> None:
        safe = {
            key: value
            for key, value in fields.items()
            if "credential" not in key.casefold()
            and "api_key" not in key.casefold()
            and "authorization" not in key.casefold()
        }
        self.logger.info(json.dumps({"event": event, **safe}, default=str, sort_keys=True))


def _health(attempts: list[ProviderAttempt], now: datetime) -> list[ProviderHealth]:
    output = []
    for provider_id in sorted({a.provider_id for a in attempts}):
        items = [a for a in attempts if a.provider_id == provider_id]
        successes = [a for a in items if a.success]
        failures = [a for a in items if not a.success]
        state = HealthState.HEALTHY if not failures else HealthState.DEGRADED if successes else HealthState.UNAVAILABLE
        output.append(
            ProviderHealth(
                provider_id=provider_id,
                state=state,
                last_success=now if successes else None,
                last_failure=now if failures else None,
                failure_count=len(failures),
                latency_ms=sum(a.latency_ms for a in items) / len(items),
                documents_returned=sum(a.documents_received for a in items),
                rate_limited=any(a.rate_limited for a in items),
            )
        )
    return output


def _source_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_intelligence_cycle(
    queries: list[QuerySpec],
    retriever: MultiProviderRetriever,
    extractor: EventExtractor,
    store: IntelligenceStore,
    configuration: object,
    forecast_origin: datetime,
    manifest_path: str | Path | None = None,
    cache: DeterministicCache | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    source_sha: str | None = None,
    logger: StructuredRunLogger | None = None,
) -> IntelligenceRunReport:
    started, run_id, logger = now(), uuid.uuid4().hex, logger or StructuredRunLogger()
    fingerprint = configuration_fingerprint(configuration)
    logger.emit("run_started", run_id=run_id, queries_attempted=len(queries))
    retrieval = retriever.retrieve(queries)
    documents = deduplicate_across_providers(retrieval.documents)
    cache_hits, cache_misses = 0, 0
    if cache is not None:
        key = hashlib.sha256(
            json.dumps(
                {
                    "queries": [q.model_dump(mode="json") for q in queries],
                    "providers": sorted(retriever.providers),
                    "extractor": extractor.version,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        cached = cache.get(key)
        if cached is None:
            cache.put(key, [d.model_dump(mode="json") for d in documents])
            cache_misses = 1
        else:
            documents = [Document.model_validate(d) for d in cached]
            cache_hits = 1
    quarantine = [
        QuarantineRecord.from_raw(
            a.error or "provider failure", a.provider_id, now(), a.error or "provider failure", "PROVIDER_FAILURE"
        )
        for a in retrieval.attempts
        if not a.success
    ]
    rejected_events = 0
    try:
        events = extractor.extract(documents)
    except Exception as exc:
        events, rejected_events = [], 1
        quarantine.append(
            QuarantineRecord.from_raw(str(exc), "extractor", now(), type(exc).__name__, "BAD_EXTRACTOR_OUTPUT")
        )
    quality = evaluate_quality(documents, events, as_of=forecast_origin)
    watermarks = []
    query_by_id = {q.query_id: q for q in queries}
    for attempt in retrieval.attempts:
        planned_query = query_by_id[attempt.query_id].query
        related = [d for d in documents if d.provider == attempt.provider_id and d.query == planned_query]
        latest = max(related, key=lambda d: d.available_at) if related else None
        watermarks.append(
            Watermark(
                provider_id=attempt.provider_id,
                query_id=attempt.query_id,
                last_successful_available_time=latest.available_at if latest and attempt.success else None,
                last_retrieval_time=now(),
                last_document_id=latest.document_id if latest else None,
                status=WatermarkStatus.SUCCESS if attempt.success else WatermarkStatus.FAILED,
            )
        )
    store.persist_cycle(documents, events, watermarks)
    store.put_quarantine(quarantine)
    features = FeatureAggregator().aggregate(events, forecast_origin)
    successful, failed = retrieval.queries_successful, retrieval.queries_failed
    missingness = MissingnessReport(
        provider_available=successful > 0,
        coverage_ratio=successful / len(retrieval.attempts) if retrieval.attempts else 0.0,
        documents_seen=len(documents),
        queries_successful=successful,
        queries_failed=failed,
    )
    finished = now()
    provider_health = _health(retrieval.attempts, finished)
    store.put_health(provider_health)
    status = (
        RunStatus.FAILED
        if successful == 0
        else RunStatus.PARTIAL_SUCCESS
        if failed
        else RunStatus.DEGRADED
        if not quality.valid or rejected_events > 0
        else RunStatus.SUCCESS
    )
    manifest = RunManifest(
        run_id=run_id,
        started_at=started,
        finished_at=finished,
        configuration_fingerprint=fingerprint,
        providers_attempted=len({a.provider_id for a in retrieval.attempts}),
        queries_attempted=len(retrieval.attempts),
        documents_accepted=len(documents),
        documents_rejected=max(0, len(retrieval.documents) - len(documents)),
        events_accepted=len(events),
        events_rejected=rejected_events,
        quality_summary={
            "valid": quality.valid,
            "issues": len(quality.issues),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "provider_coverage_ratio": missingness.coverage_ratio,
            "queries_failed": missingness.queries_failed,
            "source_stale_flag": int(any(h.state == HealthState.STALE for h in provider_health)),
        },
        watermark_changes=sum(w.status == WatermarkStatus.SUCCESS for w in watermarks),
        software_source_sha=source_sha or _source_sha(),
        status=status,
        provider_ids=tuple(sorted({a.provider_id for a in retrieval.attempts})),
    )
    issue_codes = [issue.code for issue in quality.issues]
    scoreboard = QualityScoreboard(
        run_id=run_id,
        created_at=finished,
        documents_accepted=len(documents),
        duplicates_removed=max(0, len(retrieval.documents) - len(documents)),
        events_accepted=len(events),
        events_rejected=rejected_events,
        orphan_events=issue_codes.count("ORPHANED_EVENT"),
        future_availability_violations=issue_codes.count("FUTURE_AVAILABILITY"),
        invalid_schema_records=issue_codes.count("SCHEMA_VERSION"),
        quarantined_records=len(quarantine),
        provider_coverage_ratio=missingness.coverage_ratio,
        stale_providers=sum(h.state == HealthState.STALE for h in provider_health),
    )
    store.put_run(manifest)
    store.put_quality_scoreboard(scoreboard)
    store.put_provider_attempts(run_id, retrieval.attempts, finished)
    if manifest_path:
        manifest.write_atomic(manifest_path)  # deliberately last durable run artifact
    logger.emit(
        "run_finished",
        run_id=run_id,
        documents_received=len(retrieval.documents),
        documents_deduped=len(documents),
        events_created=len(events),
        events_rejected=rejected_events,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        status="PARTIAL" if failed else "SUCCESS",
    )
    return IntelligenceRunReport(
        run_id=run_id,
        documents=documents,
        events=events,
        features=features,
        attempts=retrieval.attempts,
        missingness=missingness,
        provider_health=provider_health,
        manifest=manifest,
    )


class ReplayService:
    def __init__(self, store: IntelligenceStore):
        self.store = store

    def replay(
        self, forecast_origin: datetime, provider_versions: dict[str, str], config_fingerprint: str
    ) -> IntelligenceSnapshot:
        documents = self.store.documents_as_of(forecast_origin)
        events = self.store.signals_as_of(forecast_origin)
        eligible_ids = {d.document_id for d in documents}
        events = [e for e in events if set(e.source_ids) <= eligible_ids]
        snapshot = IntelligenceSnapshot.create(
            forecast_origin,
            [d.document_id for d in documents],
            [e.event_id for e in events],
            FeatureAggregator().aggregate(events, forecast_origin),
            provider_versions,
            [e.extractor_version for e in events],
            config_fingerprint,
            [d.text_hash for d in documents],
            FEATURE_CONTRACT_VERSION,
        )
        self.store.put_snapshot(snapshot)
        return snapshot
