"""B4.1.17 / B4.1.25 — one forward collection cycle, manifested.

Wraps the existing `run_intelligence_cycle` rather than replacing it: query
planning, retrieval, dedup, extraction, quality, watermarks and atomic
persistence already work and B4.1.0 said not to replace what works. What this
adds is the forward-collection layer around it — raw evidence, clustering,
corpus snapshotting, and a manifest that records what a run actually did.

The ordering matters and mirrors B3.1's: **the manifest is written last**, after
persistence succeeded. A manifest written first would claim work that a crash
then prevented, and the watermark would advance past evidence that was never
stored — which is the one failure mode B4.1.20 exists to prevent.

Partial provider failure is not a run failure. One dead feed among eight must
not discard the other seven's documents; the failure is classified, recorded
against provider health, and the cycle completes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict

from ..configuration import QuerySpec
from ..cycle import IntelligenceRunReport, run_intelligence_cycle
from ..extractors import EventExtractor
from ..retrieval import MultiProviderRetriever
from ..storage import IntelligenceStore
from .clustering import ClusterStore, cluster_events
from .corpus import CorpusCatalog, CorpusSnapshot, build_snapshot
from .evidence import EvidenceStore, RawEvidence
from .policy import redact_mapping

COLLECTION_MANIFEST_VERSION = "b41-collection-manifest-v2"


class CollectionManifest(BaseModel):
    """B4.1.25. Immutable record of one collection run."""

    model_config = ConfigDict(frozen=True)

    manifest_version: str = COLLECTION_MANIFEST_VERSION
    run_id: str
    started_at: datetime
    finished_at: datetime
    source_sha: str
    configuration_fingerprint: str
    provider_ids: tuple[str, ...]
    provider_success: dict[str, bool]
    query_ids: tuple[str, ...]
    documents_retrieved: int
    documents_new: int
    documents_rediscovered: int
    events_extracted: int
    clusters: int
    extractor_versions: tuple[str, ...]
    raw_evidence_ids: tuple[str, ...]
    #: How many of those were new. In steady state most evidence is a repeat
    #: fetch of an unchanged feed, and the gap between the two is the signal
    #: that content-addressed storage is deduplicating rather than growing.
    raw_evidence_ids_stored: int = 0
    raw_evidence_bytes: int
    watermarks_advanced: int
    #: Records set aside for a human: a provider that failed, or an extractor
    #: that raised. Emphatically *not* deduplication -- see below.
    quarantined: int
    #: The same item arriving under several queries, which is the normal shape
    #: of a cycle rather than a fault. It was previously counted as quarantine,
    #: so a healthy run reported twelve quarantined records and an empty
    #: quarantine table. An operator reading that either investigates a
    #: non-problem or learns to distrust the number; both are worse than not
    #: reporting it.
    documents_deduplicated: int = 0
    errors: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()
    corpus_id: str | None = None

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()

    def write(self, path: str | Path) -> Path:
        """Atomic, and never over a manifest that already exists.

        A run id is unique per run, so an existing manifest at this path means
        two runs are writing to one place -- which would leave a record that
        describes neither.
        """
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"{target} already holds a collection manifest; runs never overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        payload["manifest_hash"] = self.content_hash()
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return target


class ForwardCollectionResult(BaseModel):
    """What a cycle produced, for the CLI and the API."""

    model_config = ConfigDict(frozen=True)

    manifest: CollectionManifest
    report: IntelligenceRunReport
    snapshot: CorpusSnapshot | None


class ForwardCollector:
    """Runs collection cycles and maintains the derived corpus structures."""

    def __init__(
        self,
        store: IntelligenceStore,
        *,
        cluster_window_hours: int = 24,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.store = store
        self.evidence = EvidenceStore(store.connection)
        self.clusters = ClusterStore(store.connection)
        self.catalog = CorpusCatalog(store.connection)
        self.cluster_window_hours = cluster_window_hours
        self._now = now

    def collect(
        self,
        queries: list[QuerySpec],
        retriever: MultiProviderRetriever,
        extractor: EventExtractor,
        configuration: object,
        *,
        manifest_path: str | Path | None = None,
        raw_evidence: Sequence[RawEvidence] = (),
        register_snapshot: bool = True,
        source_sha: str | None = None,
    ) -> ForwardCollectionResult:
        """Run one cycle, then rebuild the derived corpus structures.

        The origin is *now*. Forward collection has no other defensible choice:
        evidence becomes available when it is retrieved, and a cycle that
        claimed a past origin would be asserting availability it cannot show.
        """
        started = self._now()
        before = self._document_count()

        report = run_intelligence_cycle(
            queries,
            retriever,
            extractor,
            self.store,
            configuration,
            started,
            manifest_path=None,  # this layer writes the manifest, last
            now=self._now,
            source_sha=source_sha,
        )

        # Evidence is gathered *after* the cycle, not before. A caller passing
        # `provider.last_evidence` as an argument would hand over an empty list,
        # because the providers have not run at the moment the call is made --
        # a trap worth removing rather than documenting.
        collected = list(raw_evidence) + _provider_evidence(retriever)
        # Stored after the cycle persisted its documents, so a crash leaves
        # evidence without a document rather than the reverse: an orphan payload
        # is inert, a document with no evidence is a gap.
        newly_stored_evidence = self.evidence.put(collected)

        # The corpus view is taken *after* persistence, not at `started`.
        # Retrieval happens during the cycle, so this run's own documents become
        # available after it began -- reading the store as of `started` would
        # correctly exclude them and leave every snapshot one cycle behind the
        # evidence it was meant to record.
        observed_at = self._now()
        documents = self.store.documents_as_of(observed_at)
        events = self.store.signals_as_of(observed_at)
        clusters = cluster_events(events, documents, window_hours=self.cluster_window_hours)
        self.clusters.replace_all(clusters)

        snapshot: CorpusSnapshot | None = None
        if register_snapshot and documents:
            candidate = build_snapshot(
                documents,
                events,
                as_of=observed_at,
                extractor_version=extractor.version,
                created_at=observed_at,
                cluster_window_hours=self.cluster_window_hours,
            )
            # Register only when the corpus actually changed. A snapshot id
            # includes its as-of instant, so an hourly cycle would otherwise
            # append a distinct-but-identical entry every hour -- 8,760 a year
            # of noise the catalog exists to be free of. The membership hash is
            # what makes "changed" answerable.
            latest = self.catalog.latest()
            if latest is None or latest.membership_hash != candidate.membership_hash:
                snapshot = self.catalog.register(candidate)
            else:
                snapshot = latest

        after = self._document_count()
        finished = self._now()

        retrieved = len(report.documents)
        manifest = CollectionManifest(
            run_id=report.run_id,
            started_at=started,
            finished_at=finished,
            source_sha=source_sha or report.manifest.software_source_sha,
            configuration_fingerprint=report.manifest.configuration_fingerprint,
            provider_ids=tuple(sorted({attempt.provider_id for attempt in report.attempts})),
            provider_success={attempt.provider_id: bool(attempt.success) for attempt in report.attempts},
            query_ids=tuple(sorted(query.query_id for query in queries)),
            documents_retrieved=retrieved,
            documents_new=after - before,
            # Everything retrieved that was already in the store. Under forward
            # collection this is the *majority* of a steady-state cycle -- feeds
            # re-serve the same items -- and it is the number that shows the
            # first-write-wins rule doing its job.
            documents_rediscovered=max(0, retrieved - (after - before)),
            events_extracted=len(report.events),
            clusters=len(clusters),
            extractor_versions=tuple(sorted({event.extractor_version for event in events})),
            raw_evidence_ids=tuple(sorted(record.evidence_id for record in collected)),
            raw_evidence_ids_stored=newly_stored_evidence,
            raw_evidence_bytes=sum(record.content_bytes for record in collected),
            watermarks_advanced=report.manifest.watermark_changes,
            # `documents_rejected` is computed as len(retrieved) - len(deduplicated),
            # which is the duplicate count and nothing else. Only failed attempts
            # and a raising extractor actually reach the quarantine table.
            quarantined=(
                sum(1 for attempt in report.attempts if not attempt.success)
                + report.manifest.events_rejected
            ),
            documents_deduplicated=report.manifest.documents_rejected,
            errors=tuple(
                sorted(
                    f"{attempt.provider_id}: {attempt.error}"
                    for attempt in report.attempts
                    if not attempt.success and attempt.error
                )
            ),
            quality_flags=tuple(sorted(str(key) for key in report.manifest.quality_summary)),
            corpus_id=snapshot.corpus_id if snapshot else None,
        )

        if manifest_path is not None:
            manifest.write(manifest_path)
        return ForwardCollectionResult(manifest=manifest, report=report, snapshot=snapshot)

    def successful_run_days(self) -> list[datetime]:
        """Days on which at least one provider succeeded (B4.1.23)."""
        rows = self.store.connection.execute(
            "SELECT payload FROM watermarks"
        ).fetchall()
        days: list[datetime] = []
        for (payload,) in rows:
            record = json.loads(payload)
            moment = record.get("last_retrieval_time")
            if moment:
                days.append(datetime.fromisoformat(moment).astimezone(timezone.utc))
        return days

    def _document_count(self) -> int:
        row = self.store.connection.execute("SELECT count(*) FROM documents").fetchone()
        return int(row[0]) if row else 0


def _provider_evidence(retriever: MultiProviderRetriever) -> list[RawEvidence]:
    """Collect raw evidence from any provider that captured some.

    Duck-typed on purpose: a provider that retains nothing simply has no
    `last_evidence`, and requiring every adapter to implement an evidence
    protocol would make the optional feature mandatory.
    """
    records: list[RawEvidence] = []
    for provider in retriever.providers.values():
        found = getattr(provider, "last_evidence", None)
        if found:
            records.extend(found)
    return records


def cadence_due(
    last_run: datetime | None, minimum_interval_seconds: int, now: datetime
) -> bool:
    """B4.1.19. Whether a provider may be polled again yet.

    Never polling faster than the declared interval is the whole mechanism. A
    provider with no recorded run is due -- a first run has nothing to be too
    soon after.
    """
    if last_run is None:
        return True
    return now - last_run >= timedelta(seconds=minimum_interval_seconds)


def redacted_configuration(settings: dict[str, Any]) -> dict[str, str]:
    """B4.1.40. Configuration safe to put in a manifest or an API response."""
    return redact_mapping({key: str(value) for key, value in settings.items()})


__all__ = [
    "COLLECTION_MANIFEST_VERSION",
    "CollectionManifest",
    "ForwardCollectionResult",
    "ForwardCollector",
    "cadence_due",
    "redacted_configuration",
]
