"""O3 / O4 / O6 — the one command a scheduler calls, and what it guarantees.

`btc-intel collect-scheduled` is the whole operational surface. It takes the run
lock, validates persistent storage, decides which providers are due, runs one
cycle, writes the manifest last, releases the lock, and exits with a code the
scheduler can branch on.

The ordering is the contract, and it is the same one B3.1 established:

1. lock  — so two schedulers cannot interleave
2. storage validation — so a cycle never runs where it cannot persist
3. cadence — so no provider is polled faster than it declared
4. cycle — retrieval, dedup, extraction, persistence, watermarks
5. manifest — **last**, because its presence is the claim that everything before
   it finished

A crash anywhere leaves no manifest, an unadvanced watermark for the provider
that did not complete, and a lock that the next run breaks as stale. That is the
whole recovery story, and it needs no separate recovery mode: the next scheduled
run is the recovery.

Exit codes: 0 ran, 3 skipped because another collector holds the lock (not a
failure — the scheduler fired while a long cycle was still going), 4 nothing was
due, 2 a real failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..collection.policy import ProviderDeclaration
from ..collection.service import ForwardCollectionResult, ForwardCollector, cadence_due
from ..configuration import QuerySpec
from ..extractors import EventExtractor
from ..retrieval import MultiProviderRetriever
from ..storage import IntelligenceStore
from .paths import StoragePaths, looks_ephemeral, require_usable
from .runlock import LockHeld, RunLock

EXIT_OK = 0
EXIT_FAILED = 2
EXIT_LOCK_HELD = 3
EXIT_NOTHING_DUE = 4


@dataclass(frozen=True)
class ScheduledOutcome:
    """What one scheduled invocation did."""

    exit_code: int
    reason: str
    ran: bool
    manifest_path: Path | None = None
    result: ForwardCollectionResult | None = None
    due_providers: tuple[str, ...] = ()
    skipped_providers: tuple[str, ...] = ()
    stale_lock_broken: bool = False
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exit_code": self.exit_code,
            "reason": self.reason,
            "ran": self.ran,
            "due_providers": list(self.due_providers),
            "skipped_providers": list(self.skipped_providers),
            "stale_lock_broken": self.stale_lock_broken,
            "warnings": list(self.warnings),
        }
        if self.manifest_path:
            payload["manifest"] = str(self.manifest_path)
        if self.result is not None:
            manifest = self.result.manifest
            payload["collection"] = {
                "run_id": manifest.run_id,
                "documents_retrieved": manifest.documents_retrieved,
                "documents_new": manifest.documents_new,
                "documents_rediscovered": manifest.documents_rediscovered,
                "events_extracted": manifest.events_extracted,
                "clusters": manifest.clusters,
                "corpus_id": manifest.corpus_id,
                "provider_success": manifest.provider_success,
            }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def due_providers(
    declarations: Mapping[str, ProviderDeclaration],
    last_success: Mapping[str, datetime | None],
    now: datetime,
) -> tuple[list[str], list[str]]:
    """O4. Split providers into those due to be polled and those that are not.

    Cadence is a floor this system imposes on itself, not the provider's limit.
    A provider polled ten minutes ago is skipped even if its documented quota
    would allow it -- research need, not maximum extraction.
    """
    due: list[str] = []
    skipped: list[str] = []
    for provider_id in sorted(declarations):
        declaration = declarations[provider_id]
        if cadence_due(last_success.get(provider_id), declaration.minimum_interval_seconds, now):
            due.append(provider_id)
        else:
            skipped.append(provider_id)
    return due, skipped


def run_scheduled(
    paths: StoragePaths,
    *,
    build_queries: Callable[[datetime], list[QuerySpec]],
    build_retriever: Callable[[Sequence[str]], MultiProviderRetriever],
    extractor: EventExtractor,
    configuration: object,
    declarations: Mapping[str, ProviderDeclaration],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    require_free_bytes: int = 64 * 1024 * 1024,
    source_sha: str | None = None,
) -> ScheduledOutcome:
    """One scheduled collection cycle, or an explained reason for not running."""
    moment = now()
    warnings: list[str] = []

    ephemeral = looks_ephemeral(paths.root)
    if ephemeral:
        warnings.append(ephemeral)

    paths.ensure()
    require_usable(paths, require_free_bytes=require_free_bytes)

    lock = RunLock(paths.lock, purpose="collect-scheduled", now=now)
    try:
        lock.acquire()
    except LockHeld as held:
        # Not a failure. A scheduler firing while a long cycle is still running
        # is normal, and treating it as an error trains operators to ignore
        # the collector's exit code.
        return ScheduledOutcome(
            exit_code=EXIT_LOCK_HELD,
            reason=str(held),
            ran=False,
            warnings=tuple(warnings),
        )

    try:
        store = IntelligenceStore(paths.database)
        try:
            collector = ForwardCollector(store, now=now)
            last_success = _last_success_by_provider(store)
            due, skipped = due_providers(declarations, last_success, moment)
            if not due:
                return ScheduledOutcome(
                    exit_code=EXIT_NOTHING_DUE,
                    reason="no provider is due yet under its declared cadence",
                    ran=False,
                    skipped_providers=tuple(skipped),
                    stale_lock_broken=lock.broke_stale_lock,
                    warnings=tuple(warnings),
                )

            queries = build_queries(moment)
            retriever = build_retriever(due)
            manifest_path = paths.manifests / f"collect-{moment.strftime('%Y%m%dT%H%M%SZ')}.json"
            result = collector.collect(
                queries,
                retriever,
                extractor,
                configuration,
                manifest_path=manifest_path,
                source_sha=source_sha,
            )
            return ScheduledOutcome(
                exit_code=EXIT_OK,
                reason="collection cycle completed",
                ran=True,
                manifest_path=manifest_path,
                result=result,
                due_providers=tuple(due),
                skipped_providers=tuple(skipped),
                stale_lock_broken=lock.broke_stale_lock,
                warnings=tuple(warnings),
            )
        finally:
            store.close()
    finally:
        lock.release()


def _last_success_by_provider(store: IntelligenceStore) -> dict[str, datetime | None]:
    """Most recent successful retrieval per provider, from the watermarks.

    Watermarks are the record of what actually completed, which is what cadence
    should be measured from -- an attempt that failed did not consume the
    provider's quota in any way that matters to research.
    """
    rows = store.connection.execute("SELECT payload FROM watermarks").fetchall()
    latest: dict[str, datetime | None] = {}
    for (payload,) in rows:
        record = json.loads(payload)
        provider = str(record.get("provider_id", ""))
        retrieval = record.get("last_retrieval_time")
        if not provider or not retrieval:
            continue
        moment = datetime.fromisoformat(retrieval).astimezone(timezone.utc)
        current = latest.get(provider)
        if current is None or moment > current:
            latest[provider] = moment
    return latest


def last_run_times(store: IntelligenceStore) -> tuple[datetime | None, datetime | None]:
    """(last attempt, last success) across all providers, for the watchdog."""
    latest = _last_success_by_provider(store)
    successes = [moment for moment in latest.values() if moment is not None]
    newest = max(successes) if successes else None
    return newest, newest


__all__ = [
    "EXIT_FAILED",
    "EXIT_LOCK_HELD",
    "EXIT_NOTHING_DUE",
    "EXIT_OK",
    "ScheduledOutcome",
    "due_providers",
    "last_run_times",
    "run_scheduled",
]
