"""O9 / O10 — is collection actually working, and a vendor-neutral way to say so.

The distinction this module exists to protect, again: **a quiet day is not a
collection failure.** Regulator feeds genuinely produce nothing bitcoin-related
most days. A watchdog that alerts on "zero documents today" would fire
constantly, be muted within a week, and then be silent on the day collection
actually stops — which is the only day it mattered.

So the signal is never the document count on its own. It is whether providers
*succeeded*. Zero documents with every provider healthy is a quiet day; zero
documents with every provider failing is an outage; and the implausible case —
a provider reporting success while returning nothing for days on end — gets its
own alert, because that is what the four B4.1 silent-empty defects looked like
from the outside.

Alerts are plain objects with a severity and a machine-readable code. No vendor
is wired in. JSON on stdout is a complete integration: cron mails it, a systemd
timer journals it, a container ships it. Anything more opinionated would be a
second system to operate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from ..collection.status import CorpusStatus
from .integrity import IntegrityStatus


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCode(str, Enum):
    """O10. Stable codes, so a downstream rule never matches on prose."""

    COLLECTION_STALE = "COLLECTION_STALE"
    PROVIDER_STALE = "PROVIDER_STALE"
    ALL_PROVIDERS_FAILED = "ALL_PROVIDERS_FAILED"
    STORAGE_FAILURE = "STORAGE_FAILURE"
    CORPUS_INTEGRITY_FAILURE = "CORPUS_INTEGRITY_FAILURE"
    BACKUP_FAILURE = "BACKUP_FAILURE"
    BACKUP_STALE = "BACKUP_STALE"
    QUARANTINE_SPIKE = "QUARANTINE_SPIKE"
    IMPLAUSIBLE_ZERO_COLLECTION = "IMPLAUSIBLE_ZERO_COLLECTION"
    LOCK_STALE_BROKEN = "LOCK_STALE_BROKEN"
    DISK_LOW = "DISK_LOW"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    COLLECTION_HEALTHY = "COLLECTION_HEALTHY"


@dataclass(frozen=True)
class Alert:
    code: AlertCode
    severity: Severity
    message: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WatchdogPolicy:
    """Thresholds, declared. Generous, because a noisy watchdog is a muted one."""

    #: No successful cycle for this long is a problem worth waking someone for.
    collection_stale_after: timedelta = timedelta(hours=6)
    #: A single provider silent for this long, while others work.
    provider_stale_after: timedelta = timedelta(days=2)
    #: A provider succeeding but returning nothing for this many consecutive
    #: days. Rare enough to be worth a look, common enough on quiet feeds that
    #: it is a WARNING and not a CRITICAL.
    implausible_zero_days: int = 14
    backup_stale_after: timedelta = timedelta(days=7)
    quarantine_spike: int = 25
    #: Free space on the state root. The collector itself refuses to start below
    #: 64 MiB; these lines are set well above that so an operator hears first.
    #: The corpus grows by megabytes a month, so a warning is days of notice.
    disk_warning_free_bytes: int = 1024 * 1024 * 1024
    disk_critical_free_bytes: int = 256 * 1024 * 1024


#: Built once; a caller wanting different thresholds passes them explicitly.
DEFAULT_WATCHDOG_POLICY = WatchdogPolicy()


@dataclass(frozen=True)
class WatchdogInput:
    """Everything the assessment needs, gathered by the caller.

    Passed in rather than fetched here so the watchdog is a pure function of
    observable state -- which is what makes it testable without a store.
    """

    now: datetime
    last_run_at: datetime | None
    last_successful_run_at: datetime | None
    providers_enabled: int
    providers_healthy: int
    provider_last_success: dict[str, datetime | None]
    documents_last_24h: int
    events_last_24h: int
    quarantined_last_24h: int
    storage_ok: bool
    storage_detail: str
    integrity_status: IntegrityStatus
    integrity_detail: str
    last_backup_at: datetime | None
    consecutive_zero_days_with_success: int
    stale_lock_broken: bool = False
    #: The recorded status of the most recent run, from the runs table. Unlike
    #: provider_last_success, which remembers every past success, this says
    #: what happened last.
    last_run_status: str | None = None
    #: Free bytes on the state root; None when it could not be measured.
    free_bytes: int | None = None
    #: Why the deployed configuration would not load, if it would not.
    configuration_error: str | None = None


@dataclass(frozen=True)
class WatchdogResult:
    alerts: tuple[Alert, ...]
    healthy: bool

    @property
    def worst(self) -> Severity:
        for severity in (Severity.CRITICAL, Severity.WARNING, Severity.INFO):
            if any(alert.severity is severity for alert in self.alerts):
                return severity
        return Severity.INFO

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "worst_severity": self.worst.value,
            "alerts": [alert.as_dict() for alert in self.alerts],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(target)
        return target

    @property
    def exit_code(self) -> int:
        """0 healthy, 1 warning, 2 critical -- so cron and systemd can branch."""
        return {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}[self.worst]


def assess(state: WatchdogInput, policy: WatchdogPolicy | None = None) -> WatchdogResult:
    """Turn observable state into alerts. Deterministic, no I/O."""
    policy = policy or DEFAULT_WATCHDOG_POLICY
    alerts: list[Alert] = []

    # Storage and integrity first: neither is recoverable by waiting, and both
    # invalidate everything downstream of them.
    if not state.storage_ok:
        alerts.append(
            Alert(
                AlertCode.STORAGE_FAILURE,
                Severity.CRITICAL,
                "persistent storage is not usable; collection cannot accumulate",
                {"detail": state.storage_detail},
            )
        )
    if state.integrity_status is IntegrityStatus.CORRUPT:
        alerts.append(
            Alert(
                AlertCode.CORPUS_INTEGRITY_FAILURE,
                Severity.CRITICAL,
                "corpus integrity check failed; research on this corpus is unsafe",
                {"detail": state.integrity_detail},
            )
        )
    elif state.integrity_status is IntegrityStatus.DEGRADED:
        alerts.append(
            Alert(
                AlertCode.CORPUS_INTEGRITY_FAILURE,
                Severity.WARNING,
                "corpus integrity is degraded; no availability claim is affected",
                {"detail": state.integrity_detail},
            )
        )

    if state.configuration_error is not None:
        alerts.append(
            Alert(
                AlertCode.CONFIGURATION_INVALID,
                Severity.CRITICAL,
                "the deployed configuration does not load; the next cycle will fail before collecting",
                {"detail": state.configuration_error},
            )
        )

    if state.free_bytes is not None and state.free_bytes < policy.disk_warning_free_bytes:
        critical = state.free_bytes < policy.disk_critical_free_bytes
        alerts.append(
            Alert(
                AlertCode.DISK_LOW,
                Severity.CRITICAL if critical else Severity.WARNING,
                f"{state.free_bytes:,} bytes free on the state root",
                {
                    "free_bytes": state.free_bytes,
                    "threshold_bytes": (
                        policy.disk_critical_free_bytes if critical else policy.disk_warning_free_bytes
                    ),
                },
            )
        )

    if state.last_successful_run_at is None:
        alerts.append(
            Alert(
                AlertCode.COLLECTION_STALE,
                Severity.CRITICAL,
                "no successful collection cycle has ever completed",
                {},
            )
        )
    else:
        since = state.now - state.last_successful_run_at
        if since > policy.collection_stale_after:
            alerts.append(
                Alert(
                    AlertCode.COLLECTION_STALE,
                    Severity.CRITICAL,
                    f"no successful collection for {_human(since)}",
                    {
                        "last_successful_run_at": state.last_successful_run_at.isoformat(),
                        "threshold": _human(policy.collection_stale_after),
                    },
                )
            )

    if state.providers_enabled and state.providers_healthy == 0:
        alerts.append(
            Alert(
                AlertCode.ALL_PROVIDERS_FAILED,
                Severity.CRITICAL,
                f"all {state.providers_enabled} enabled provider(s) are failing",
                {"providers_enabled": state.providers_enabled},
            )
        )

    # B5.2. provider_last_success remembers every past success, so the check
    # above can only fire on a deployment that has never collected. The last
    # run's own status is what says every provider is failing *now*.
    if state.last_run_status == "FAILED" and not any(
        alert.code is AlertCode.ALL_PROVIDERS_FAILED for alert in alerts
    ):
        alerts.append(
            Alert(
                AlertCode.ALL_PROVIDERS_FAILED,
                Severity.CRITICAL,
                "the last collection cycle read nothing from any provider",
                {"last_run_status": state.last_run_status},
            )
        )

    for provider, last_success in sorted(state.provider_last_success.items()):
        if last_success is None:
            alerts.append(
                Alert(
                    AlertCode.PROVIDER_STALE,
                    Severity.WARNING,
                    f"provider {provider} has never succeeded",
                    {"provider": provider},
                )
            )
        elif state.now - last_success > policy.provider_stale_after:
            alerts.append(
                Alert(
                    AlertCode.PROVIDER_STALE,
                    Severity.WARNING,
                    f"provider {provider} last succeeded {_human(state.now - last_success)} ago",
                    {"provider": provider, "last_success": last_success.isoformat()},
                )
            )

    # The B4.1 failure mode, from the outside: providers reporting success while
    # returning nothing, for longer than a quiet stretch plausibly explains.
    if state.consecutive_zero_days_with_success >= policy.implausible_zero_days:
        alerts.append(
            Alert(
                AlertCode.IMPLAUSIBLE_ZERO_COLLECTION,
                Severity.WARNING,
                f"{state.consecutive_zero_days_with_success} consecutive days of successful "
                "provider calls returning no documents; check query terms and feed contents "
                "before assuming the news is quiet",
                {"days": state.consecutive_zero_days_with_success},
            )
        )

    if state.quarantined_last_24h >= policy.quarantine_spike:
        alerts.append(
            Alert(
                AlertCode.QUARANTINE_SPIKE,
                Severity.WARNING,
                f"{state.quarantined_last_24h} records quarantined in 24h",
                {"quarantined": state.quarantined_last_24h},
            )
        )

    if state.last_backup_at is None:
        alerts.append(
            Alert(AlertCode.BACKUP_STALE, Severity.WARNING, "no backup has been taken", {})
        )
    elif state.now - state.last_backup_at > policy.backup_stale_after:
        alerts.append(
            Alert(
                AlertCode.BACKUP_STALE,
                Severity.WARNING,
                f"last backup was {_human(state.now - state.last_backup_at)} ago",
                {"last_backup_at": state.last_backup_at.isoformat()},
            )
        )

    if state.stale_lock_broken:
        alerts.append(
            Alert(
                AlertCode.LOCK_STALE_BROKEN,
                Severity.INFO,
                "a stale run lock was broken; the previous cycle did not exit cleanly",
                {},
            )
        )

    if not alerts:
        alerts.append(
            Alert(
                AlertCode.COLLECTION_HEALTHY,
                Severity.INFO,
                "collection is healthy",
                {
                    "documents_last_24h": state.documents_last_24h,
                    "events_last_24h": state.events_last_24h,
                    # Stated explicitly so a zero here is never read as trouble.
                    "quiet_day": state.documents_last_24h == 0,
                },
            )
        )

    return WatchdogResult(
        alerts=tuple(alerts),
        healthy=not any(alert.severity is not Severity.INFO for alert in alerts),
    )


def consecutive_zero_days(
    documents_by_day: dict[date, int], successful_days: Sequence[date], today: date
) -> int:
    """Days ending today where a provider succeeded and nothing was collected.

    Counting backwards and stopping at the first day with a document -- or the
    first day with no successful run at all, since an outage breaks the pattern
    this is looking for rather than extending it.
    """
    successful = set(successful_days)
    count = 0
    cursor = today
    while cursor in successful and documents_by_day.get(cursor, 0) == 0:
        count += 1
        cursor -= timedelta(days=1)
    return count


def from_status(
    status: CorpusStatus,
    *,
    now: datetime,
    last_run_at: datetime | None,
    last_successful_run_at: datetime | None,
    provider_last_success: dict[str, datetime | None],
    storage_ok: bool,
    storage_detail: str,
    integrity_status: IntegrityStatus,
    integrity_detail: str,
    last_backup_at: datetime | None,
    quarantined_last_24h: int = 0,
    stale_lock_broken: bool = False,
    last_run_status: str | None = None,
    free_bytes: int | None = None,
    configuration_error: str | None = None,
) -> WatchdogInput:
    """Build watchdog input from a corpus status report.

    Reuses B4.1's coverage semantics rather than recomputing them, so "a gap"
    means the same thing to the watchdog as it does to the status command.
    """
    horizon = now - timedelta(hours=24)
    recent = [row for row in status.daily_coverage if row.day >= horizon.date()]
    documents_by_day = {row.day: row.documents for row in status.daily_coverage}
    successful_days = [row.day for row in status.daily_coverage if not row.collection_gap]

    return WatchdogInput(
        now=now,
        last_run_at=last_run_at,
        last_successful_run_at=last_successful_run_at,
        providers_enabled=len(provider_last_success),
        providers_healthy=sum(1 for value in provider_last_success.values() if value is not None),
        provider_last_success=provider_last_success,
        documents_last_24h=sum(row.documents for row in recent),
        events_last_24h=sum(row.events for row in recent),
        quarantined_last_24h=quarantined_last_24h,
        storage_ok=storage_ok,
        storage_detail=storage_detail,
        integrity_status=integrity_status,
        integrity_detail=integrity_detail,
        last_backup_at=last_backup_at,
        consecutive_zero_days_with_success=consecutive_zero_days(
            documents_by_day, successful_days, now.astimezone(timezone.utc).date()
        ),
        stale_lock_broken=stale_lock_broken,
        last_run_status=last_run_status,
        free_bytes=free_bytes,
        configuration_error=configuration_error,
    )


def _human(span: timedelta) -> str:
    total = int(span.total_seconds())
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h"
    return f"{total // 86400}d"


__all__ = [
    "DEFAULT_WATCHDOG_POLICY",
    "Alert",
    "AlertCode",
    "Severity",
    "WatchdogInput",
    "WatchdogPolicy",
    "WatchdogResult",
    "assess",
    "consecutive_zero_days",
    "from_status",
]
