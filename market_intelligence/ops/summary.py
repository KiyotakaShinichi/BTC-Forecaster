"""O13 / O14 / O20 — the daily operations summary, and what it must never say.

An operator reading one page a day should be able to answer: did collection run,
did it find anything, is anything broken, and how far off is B4.

What this page must **not** contain is any statistic relating collected events to
BTC returns. Not a mean, not a count of "positive days", nothing. B4's readiness
gate exists because a study on an inadequate corpus produces a confident wrong
answer, and a dashboard that showed a preliminary number every morning would
make ignoring the gate the path of least resistance. Until readiness says
otherwise, this page says `NOT_READY` and nothing about returns (O20).

Storage growth is measured from bytes actually stored and projected linearly,
labelled as a projection. The projection exists so an operator can size a volume,
not so anyone can optimise against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from ..collection.readiness import Readiness
from ..collection.status import CorpusStatus
from .watchdog import WatchdogResult

#: Anything matching these must never reach the summary. Asserted by a test, so
#: adding a return statistic to the dashboard fails CI rather than review.
FORBIDDEN_FIELDS = (
    "forward_return",
    "abs_return",
    "sharpe",
    "pnl",
    "equity",
    "signal_strength",
    "effect_size",
    "p_value",
    "q_value",
)


@dataclass(frozen=True)
class StorageProjection:
    """O14. Measured today, extrapolated linearly and labelled as such."""

    documents: int
    events: int
    clusters: int
    days_of_collection: int
    total_bytes: int
    bytes_per_document: float
    bytes_per_event: float
    bytes_per_cluster: float
    bytes_per_day: float
    projected_30d_bytes: int
    projected_180d_bytes: int
    projected_365d_bytes: int
    basis: str = (
        "linear extrapolation from measured bytes over the observed collection window; "
        "it assumes the current rate holds, which nothing guarantees"
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "events": self.events,
            "clusters": self.clusters,
            "days_of_collection": self.days_of_collection,
            "total_bytes": self.total_bytes,
            "bytes_per_document": round(self.bytes_per_document, 1),
            "bytes_per_event": round(self.bytes_per_event, 1),
            "bytes_per_cluster": round(self.bytes_per_cluster, 1),
            "bytes_per_day": round(self.bytes_per_day, 1),
            "projected_30d_bytes": self.projected_30d_bytes,
            "projected_180d_bytes": self.projected_180d_bytes,
            "projected_365d_bytes": self.projected_365d_bytes,
            "basis": self.basis,
        }


def project_storage(status: CorpusStatus) -> StorageProjection:
    """Bytes per unit, from what is actually stored.

    A corpus of one day cannot support a per-day rate, so the denominator is
    floored at one and the projection is honest about resting on very little.
    """
    measured = status.storage
    total = measured.total_bytes if measured else 0
    days = max(1, status.span_days)
    per_day = total / days
    return StorageProjection(
        documents=status.documents,
        events=status.events,
        clusters=status.clusters,
        days_of_collection=status.span_days,
        total_bytes=total,
        bytes_per_document=total / status.documents if status.documents else 0.0,
        bytes_per_event=total / status.events if status.events else 0.0,
        bytes_per_cluster=total / status.clusters if status.clusters else 0.0,
        bytes_per_day=per_day,
        projected_30d_bytes=int(per_day * 30),
        projected_180d_bytes=int(per_day * 180),
        projected_365d_bytes=int(per_day * 365),
    )


@dataclass(frozen=True)
class DailySummary:
    """O13. One deterministic page per day."""

    day: date
    generated_at: datetime
    new_documents: int
    rediscoveries: int
    new_clusters: int
    events: int
    provider_successes: dict[str, int]
    provider_failures: dict[str, int]
    coverage_ratio: float
    collection_gap: bool
    quarantined: int
    corpus_span_days: int
    corpus_documents: int
    corpus_events: int
    corpus_clusters: int
    readiness: Readiness
    families_ready: int
    families_total: int
    nearest_unmet: tuple[str, ...]
    storage: StorageProjection
    watchdog_severity: str
    alerts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "collection": {
                "new_documents": self.new_documents,
                "rediscoveries": self.rediscoveries,
                "new_clusters": self.new_clusters,
                "events": self.events,
                "provider_successes": self.provider_successes,
                "provider_failures": self.provider_failures,
                "coverage_ratio": round(self.coverage_ratio, 3),
                # Stated as its own field so a zero-document day is never read
                # as a failure, nor a failure as a quiet day.
                "collection_gap": self.collection_gap,
                "quarantined": self.quarantined,
            },
            "corpus": {
                "span_days": self.corpus_span_days,
                "documents": self.corpus_documents,
                "events": self.corpus_events,
                "clusters": self.corpus_clusters,
            },
            "b4_readiness": {
                "status": self.readiness.value,
                "families_ready": self.families_ready,
                "families_total": self.families_total,
                "nearest_unmet": list(self.nearest_unmet),
                "note": (
                    "no return statistics are reported until readiness is "
                    "READY_FOR_VALIDATION; a preliminary number would make ignoring "
                    "the gate the path of least resistance"
                ),
            },
            "storage": self.storage.as_dict(),
            "health": {"severity": self.watchdog_severity, "alerts": list(self.alerts)},
        }

    def human_readable(self) -> str:
        lines = [
            f"{self.day.isoformat()}  collection summary",
            f"  documents  +{self.new_documents} new, {self.rediscoveries} rediscovered",
            f"  events     {self.events}   clusters +{self.new_clusters}",
            f"  providers  {sum(self.provider_successes.values())} ok / "
            f"{sum(self.provider_failures.values())} failed   coverage {self.coverage_ratio:.0%}"
            + ("   [COLLECTION GAP]" if self.collection_gap else "   (quiet day)" if not self.new_documents else ""),
            f"  corpus     {self.corpus_documents} docs, {self.corpus_events} events, "
            f"{self.corpus_clusters} clusters over {self.corpus_span_days} days",
            f"  storage    {self.storage.total_bytes:,} B now, "
            f"~{self.storage.projected_365d_bytes:,} B at 365d (linear estimate)",
            f"  health     {self.watchdog_severity}",
        ]
        for alert in self.alerts:
            lines.append(f"               - {alert}")
        lines.append(
            f"  B4         {self.readiness.value}  "
            f"({self.families_ready}/{self.families_total} families ready)"
        )
        for reason in self.nearest_unmet:
            lines.append(f"               - {reason}")
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return target


def build_daily_summary(
    status: CorpusStatus,
    watchdog: WatchdogResult,
    *,
    day: date,
    generated_at: datetime,
    new_documents: int = 0,
    rediscoveries: int = 0,
    new_clusters: int = 0,
    provider_successes: dict[str, int] | None = None,
    provider_failures: dict[str, int] | None = None,
    quarantined: int = 0,
) -> DailySummary:
    """Assemble the page from the status report and the watchdog verdict."""
    row = next((item for item in status.daily_coverage if item.day == day), None)
    ready = [family for family in status.families if family.ready]

    # The clauses closest to being met, so an operator sees progress rather than
    # an undifferentiated wall of "not ready".
    unmet: list[str] = []
    for family in status.families:
        for reason in family.unmet[:1]:
            unmet.append(f"{family.family}: {reason}")

    return DailySummary(
        day=day,
        generated_at=generated_at.astimezone(timezone.utc),
        new_documents=new_documents,
        rediscoveries=rediscoveries,
        new_clusters=new_clusters,
        events=row.events if row else 0,
        provider_successes=dict(provider_successes or {}),
        provider_failures=dict(provider_failures or {}),
        coverage_ratio=row.coverage_ratio if row else 0.0,
        collection_gap=row.collection_gap if row else True,
        quarantined=quarantined,
        corpus_span_days=status.span_days,
        corpus_documents=status.documents,
        corpus_events=status.events,
        corpus_clusters=status.clusters,
        readiness=status.readiness,
        families_ready=len(ready),
        families_total=len(status.families),
        nearest_unmet=tuple(sorted(unmet)[:5]),
        storage=project_storage(status),
        watchdog_severity=watchdog.worst.value,
        alerts=tuple(alert.message for alert in watchdog.alerts),
    )


def contains_forbidden_statistics(payload: Any) -> tuple[str, ...]:
    """O20. Find any return-linked statistic that leaked into a summary.

    Used by a test rather than at runtime: the point is that adding one fails
    CI, not that it is stripped silently on the way out.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str).casefold()
    return tuple(field for field in FORBIDDEN_FIELDS if field in serialized)


def summaries_for(directory: Path, limit: int = 30) -> list[dict[str, Any]]:
    files = sorted(Path(directory).glob("summary-*.json"))[-limit:]
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def summary_path(directory: Path, day: date) -> Path:
    return Path(directory) / f"summary-{day.isoformat()}.json"


def days_between(start: datetime, end: datetime) -> Sequence[date]:
    days: list[date] = []
    cursor = start.date()
    while cursor <= end.date():
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


__all__ = [
    "FORBIDDEN_FIELDS",
    "DailySummary",
    "StorageProjection",
    "build_daily_summary",
    "contains_forbidden_statistics",
    "days_between",
    "project_storage",
    "summaries_for",
    "summary_path",
]
