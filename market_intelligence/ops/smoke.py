"""B5.2 -- the bounded smoke test a freshly deployed host runs once, and after every upgrade.

Deployment files existing is not a deployment. This runs one real collection
cycle on the host -- respecting the cadence and the run lock like any other --
and then checks, against the store, everything a first cycle is supposed to
leave behind: a recorded run, a provider that answered, documents and events,
confidences that vary with their evidence, point-in-time timing that holds, a
watermark that moved, a backup that restores, a restart that finds its place,
and a health check that is not critical.

Each check is PASS, FAIL or NOT_OBSERVED. NOT_OBSERVED is not a soft failure:
it is what a correct collector reports on a quiet week, when every feed was
read and nothing yet matched the watchlist. Only FAIL fails the smoke test, so a
deployment is never declared broken for an absence of news, and never declared
working on the strength of files alone.

It is bounded: one cycle and one backup. The restart check reopens the store
and, after a successful run, starts one more cycle that must poll nothing,
because the watermark survived; after a failed run it asks the scheduler
whether the provider is still due rather than polling again. It does not
loop, does not retry, and does not poll faster than the declared cadence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ..extractors import CURRENT_RULE_EXTRACTOR_VERSION
from ..storage import IntelligenceStore, queries
from .backup import BackupError, create_backup, verify_backup, write_backup_status, write_checksum
from .integrity import IntegrityStatus
from .integrity import verify as verify_integrity
from .paths import StoragePaths
from .paths import validate as validate_storage
from .profile import CollectionProfile, collect_once
from .runlock import LockHeld, RunLock
from .scheduled import EXIT_LOCK_HELD, EXIT_NOTHING_DUE, EXIT_OK, _last_success_by_provider, due_providers


class Outcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    #: What a correct collector reports when the evidence does not exist yet.
    NOT_OBSERVED = "NOT_OBSERVED"


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    outcome: Outcome
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "outcome": self.outcome.value, "detail": self.detail}


@dataclass(frozen=True)
class SmokeReport:
    started_at: datetime
    checks: tuple[SmokeCheck, ...]

    @property
    def ok(self) -> bool:
        return not any(check.outcome is Outcome.FAIL for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
        }

    def human_readable(self) -> str:
        lines = [f"smoke test {'passed' if self.ok else 'FAILED'} ({self.started_at.isoformat()})"]
        lines.extend(f"  {check.outcome.value:<12} {check.name:<18} {check.detail}" for check in self.checks)
        return "\n".join(lines)


def _attempts(store: IntelligenceStore, run_id: str) -> list[tuple[str, bool]]:
    rows = store.connection.execute(
        "SELECT provider_id, success FROM provider_attempts WHERE run_id = ?", [run_id]
    ).fetchall()
    return [(str(provider), bool(success)) for provider, success in rows]


def _newest_watermark(store: IntelligenceStore) -> datetime | None:
    newest: datetime | None = None
    for (payload,) in store.connection.execute("SELECT payload FROM watermarks").fetchall():
        moment = json.loads(payload).get("last_retrieval_time")
        if moment:
            parsed = datetime.fromisoformat(moment).astimezone(timezone.utc)
            newest = parsed if newest is None or parsed > newest else newest
    return newest


def _counts(store: IntelligenceStore) -> tuple[int, int]:
    documents = store.connection.execute("SELECT count(*) FROM documents").fetchone()
    events = store.connection.execute("SELECT count(*) FROM signals").fetchone()
    return (int(documents[0]) if documents else 0, int(events[0]) if events else 0)


def run_smoke(
    paths: StoragePaths,
    profile: CollectionProfile,
    *,
    profile_path: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    require_free_bytes: int = 64 * 1024 * 1024,
    source_sha: str | None = None,
    backup: bool = True,
) -> SmokeReport:
    """One bounded pass over everything a deployed collector must be able to do."""
    started = now()
    checks: list[SmokeCheck] = []

    def add(name: str, outcome: Outcome, detail: str) -> None:
        checks.append(SmokeCheck(name, outcome, detail))

    add(
        "configuration",
        Outcome.PASS if profile.user_agent else Outcome.FAIL,
        f"profile {profile.name}, {len(profile.feeds)} feed(s), "
        + ("contact User-Agent configured" if profile.user_agent else "no contact User-Agent; the SEC answers 403"),
    )
    storage = validate_storage(paths, require_free_bytes=require_free_bytes)
    add(
        "storage",
        Outcome.PASS if storage.ok else Outcome.FAIL,
        f"{paths.root} usable, {storage.free_bytes:,} bytes free"
        if storage.ok and storage.free_bytes is not None
        else "; ".join(f"{check.name}: {check.reason}" for check in storage.failures()) or str(paths.root),
    )
    if not storage.ok:
        return SmokeReport(started, tuple(checks))

    # -- one real cycle, under the lock and the cadence like any other
    cycle = collect_once(paths, profile, now=now, require_free_bytes=require_free_bytes, source_sha=source_sha)
    if cycle.exit_code == EXIT_OK:
        add("cycle", Outcome.PASS, f"one cycle ran: {cycle.run_status}")
    elif cycle.exit_code == EXIT_NOTHING_DUE:
        add("cycle", Outcome.PASS, "no provider is due yet; the most recent run is checked instead")
    elif cycle.exit_code == EXIT_LOCK_HELD:
        add("cycle", Outcome.NOT_OBSERVED, "another cycle holds the lock; run the smoke test again when it finishes")
    else:
        add("cycle", Outcome.FAIL, cycle.reason)

    # -- what the cycle left behind
    moment = now()
    store = IntelligenceStore(paths.database)
    try:
        latest = queries.latest_run_as_of(store.connection, moment + timedelta(seconds=1))
        latest_failed = latest is not None and latest.status.value == "FAILED"
        if latest is None:
            add("run_recorded", Outcome.FAIL, "no collection run is recorded")
            add("provider_request", Outcome.FAIL, "no run, so no provider was asked")
        else:
            add(
                "run_recorded",
                Outcome.FAIL if latest_failed else Outcome.PASS,
                f"run {latest.run_id[:12]} {latest.status.value}, finished {latest.finished_at.isoformat()}",
            )
            attempts = _attempts(store, latest.run_id)
            succeeded = sum(1 for _, success in attempts if success)
            add(
                "provider_request",
                Outcome.PASS if succeeded else Outcome.FAIL,
                f"{succeeded} of {len(attempts)} provider attempt(s) succeeded",
            )

        documents = store.documents_as_of(moment)
        add(
            "documents_stored",
            Outcome.PASS if documents else Outcome.NOT_OBSERVED,
            f"{len(documents)} document(s) in the corpus"
            if documents
            else "the feeds were read and nothing has matched the watchlist yet; a quiet week is not a fault",
        )
        events = [event for event in store.signals_as_of(moment) if event.extractor_version == CURRENT_RULE_EXTRACTOR_VERSION]
        add(
            "events_extracted",
            Outcome.PASS if events else Outcome.NOT_OBSERVED,
            f"{len(events)} {CURRENT_RULE_EXTRACTOR_VERSION} event(s)"
            if events
            else f"no {CURRENT_RULE_EXTRACTOR_VERSION} event yet",
        )
        confidences = sorted({round(event.confidence, 4) for event in events})
        add(
            "confidence_varies",
            Outcome.PASS if len(confidences) >= 2 else Outcome.NOT_OBSERVED,
            f"{len(confidences)} distinct confidence(s), {confidences[0]} to {confidences[-1]}"
            if len(confidences) >= 2
            else f"{len(events)} event(s); variation needs two differently evidenced events",
        )

        integrity = verify_integrity(store, as_of=moment)
        impossible = [e for e in events if e.event_time > e.available_time or e.available_time > moment]
        add(
            "point_in_time",
            Outcome.FAIL if integrity.status is IntegrityStatus.CORRUPT or impossible else Outcome.PASS,
            f"integrity {integrity.status.value}; {len(impossible)} event(s) with impossible timing",
        )

        newest = _newest_watermark(store)
        if latest is None:
            add("watermark", Outcome.FAIL, "no run to have advanced it")
        elif latest_failed:
            # A run that read nothing must not have recorded anything as collected.
            advanced = newest is not None and newest >= latest.started_at - timedelta(seconds=1)
            add(
                "watermark",
                Outcome.FAIL if advanced else Outcome.PASS,
                "a failed run advanced the watermark" if advanced else "not advanced by the failed run, as required",
            )
        else:
            advanced = newest is not None and newest >= latest.started_at - timedelta(seconds=1)
            add(
                "watermark",
                Outcome.PASS if advanced else Outcome.FAIL,
                f"advanced to {newest.isoformat()}" if advanced and newest else "the last successful run did not advance it",
            )
        counts_before = _counts(store)
    finally:
        store.close()

    # -- a backup, proved by restoring it
    if backup:
        lock = RunLock(paths.lock, purpose="smoke-backup", now=now)
        try:
            lock.acquire()
        except LockHeld:
            add("backup", Outcome.NOT_OBSERVED, "a cycle holds the lock; run `btc-intel ops-backup` afterwards")
        else:
            archive = paths.backups / f"corpus-{now().strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
            manifest = None
            try:
                live = IntelligenceStore(paths.database)
                try:
                    manifest = create_backup(live, paths.database, archive, manifest_dir=paths.manifests, now=now())
                finally:
                    live.close()
            except BackupError as error:
                add("backup", Outcome.FAIL, str(error))
            finally:
                lock.release()
            if manifest is not None:
                write_checksum(archive)
                verification = verify_backup(archive)
                write_backup_status(
                    paths.backups,
                    {
                        "at": manifest.created_at.isoformat(),
                        "archive": archive.name,
                        "ok": verification.ok,
                        "findings": list(verification.findings),
                        "sha256": verification.sha256,
                        "documents": manifest.documents,
                        "events": manifest.events,
                        "content_fingerprint": manifest.content_fingerprint,
                        "integrity_status": verification.integrity_status,
                        "source": "ops-smoke",
                    },
                )
                add("backup", Outcome.PASS, f"{archive.name}, {manifest.documents} documents, {manifest.events} events")
                restored = verification.restore
                add(
                    "restore",
                    Outcome.PASS if verification.ok else Outcome.FAIL,
                    f"restored into a scratch location: counts match, fingerprint "
                    f"{'matches' if restored is not None and restored.fingerprint_matches else 'unchecked'}, "
                    f"integrity {verification.integrity_status}"
                    if verification.ok
                    else "; ".join(verification.findings),
                )

    # -- a restart finds its place: the store reopens whole, the lock is free,
    #    and an immediate second cycle polls nothing -- unless the last run
    #    failed, in which case the provider is still due, as it must be.
    reopened = IntelligenceStore(paths.database)
    try:
        counts_after = _counts(reopened)
    finally:
        reopened.close()
    lock_free = RunLock(paths.lock).read() is None
    whole = counts_after == counts_before and lock_free
    reopened_note = (
        f"reopened with {counts_after[0]} documents and {counts_after[1]} events; "
        f"lock {'released' if lock_free else 'STILL HELD'}"
    )
    if latest_failed:
        # Asked rather than polled. A second cycle now would repeat every request
        # to the publishers, and within the same second collide with the first
        # cycle's manifest name. The scheduler's own question is enough.
        asked = IntelligenceStore(paths.database)
        try:
            due, _ = due_providers(profile.declarations(), _last_success_by_provider(asked), now())
        finally:
            asked.close()
        add(
            "restart_recovery",
            Outcome.PASS if due and whole else Outcome.FAIL,
            f"{reopened_note}; the provider is still due, as a failed run requires"
            if due
            else f"{reopened_note}; the failed run left the provider NOT due -- it recorded something as collected",
        )
    else:
        again = collect_once(paths, profile, now=now, require_free_bytes=require_free_bytes, source_sha=source_sha)
        if again.exit_code == EXIT_LOCK_HELD:
            add("restart_recovery", Outcome.NOT_OBSERVED, "another cycle holds the lock")
        else:
            polled_nothing = again.exit_code == EXIT_NOTHING_DUE
            add(
                "restart_recovery",
                Outcome.PASS if polled_nothing and whole else Outcome.FAIL,
                f"{reopened_note}; an immediate second cycle "
                + ("polled nothing: the watermark survived" if polled_nothing else f"did not stop at the cadence (exit {again.exit_code})"),
            )

    # -- and the health check is not critical
    from ..reports import ops_report

    health_store = IntelligenceStore(paths.database)
    try:
        health = ops_report(health_store, paths.database, str(paths.root), profile_path)
    finally:
        health_store.close()
    critical = [alert["code"] for alert in health["watchdog"]["alerts"] if alert["severity"] == "CRITICAL"]
    warnings = [alert["code"] for alert in health["watchdog"]["alerts"] if alert["severity"] == "WARNING"]
    add(
        "health",
        Outcome.FAIL if critical else Outcome.PASS,
        f"critical: {', '.join(critical)}" if critical else f"not critical; warnings: {', '.join(warnings) or 'none'}",
    )
    return SmokeReport(started, tuple(checks))


__all__ = ["Outcome", "SmokeCheck", "SmokeReport", "run_smoke"]
