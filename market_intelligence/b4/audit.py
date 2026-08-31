"""B4.0 / B4.33 / B4.34 — data availability audit, missingness and coverage bias.

The inventory is derived from the artifacts themselves — a store is counted, a
series is measured, a provider config is read — rather than written by hand.
A hand-written inventory is a statement of intent; this one is a measurement,
and it is allowed to come back empty.

That matters here more than usual, because an empty answer is a real possible
outcome for most of B4's categories, and the whole track depends on not papering
over it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from ..storage import IntelligenceStore
from .contracts import (
    DataAvailability,
    EvidenceTier,
    MarketSeries,
    SourceDomain,
    require_utc,
)


class SourceInventoryEntry(BaseModel):
    """One row of the B4.0 inventory."""

    model_config = ConfigDict(frozen=True)

    source: str
    domain: SourceDomain
    data_type: str
    start_date: datetime | None
    end_date: datetime | None
    frequency: str
    point_in_time_availability: str
    provenance: str
    license_status: str
    missingness: str
    observation_count: int
    availability: DataAvailability
    evidence_tier: EvidenceTier | None
    suitable_for_historical_study: bool
    notes: str = ""

    def as_record(self) -> dict[str, Any]:
        record = self.model_dump(mode="json")
        record["suitable_for_historical_study"] = "YES" if self.suitable_for_historical_study else "NO"
        return record


def inventory_market_series(
    series: MarketSeries,
    *,
    data_type: str,
    license_status: str,
    notes: str = "",
) -> SourceInventoryEntry:
    """Inventory row derived from a fetched series, not from expectation."""
    period = timedelta(seconds=series.period_seconds) if series.bars else None
    expected = 0
    if series.start is not None and series.end is not None and period:
        expected = max(1, int((series.end - series.start) / period))
    gaps = max(0, expected - len(series))
    missingness = (
        f"{gaps} of {expected} nominal periods absent ({gaps / expected:.1%})" if expected else "unknown"
    )
    if series.rejected_bar_count:
        missingness += (
            f"; {series.rejected_bar_count} rows quarantined as OHLC-incoherent "
            f"(e.g. {series.rejection_summary[0] if series.rejection_summary else 'n/a'})"
        )
    return SourceInventoryEntry(
        source=f"{series.provider}:{series.ticker}",
        domain=series.domain,
        data_type=data_type,
        start_date=series.start,
        end_date=series.end,
        frequency=f"{series.period_seconds}s bars",
        point_in_time_availability=(
            "close available at period_start + period_length (conservative; see contracts.py)"
        ),
        provenance=f"provider={series.provider} ticker={series.ticker} tz={series.source_timezone} "
        f"convention={series.timestamp_convention.value} fingerprint={series.fingerprint()[:16]}",
        license_status=license_status,
        missingness=missingness,
        observation_count=len(series),
        availability=DataAvailability.AVAILABLE if len(series) else DataAvailability.DATA_UNAVAILABLE,
        evidence_tier=series.evidence_tier,
        suitable_for_historical_study=len(series) > 0,
        notes=notes,
    )


def inventory_intelligence_store(store: IntelligenceStore) -> list[SourceInventoryEntry]:
    """Count what an intelligence store actually holds, per domain.

    Reads the store rather than the configuration on purpose: a configured
    provider that has never successfully collected anything contributes exactly
    nothing to a historical study, and only the store knows which is which.
    """
    entries: list[SourceInventoryEntry] = []

    documents = store.connection.execute(
        "SELECT count(*), min(available_at), max(available_at) FROM documents"
    ).fetchone()
    signals = store.connection.execute(
        "SELECT count(*), min(available_time), max(available_time) FROM signals"
    ).fetchone()

    entries.append(
        _corpus_entry(
            source="intelligence-store:documents",
            domain=SourceDomain.NEWS_WEB,
            data_type="retrieved web/news documents with availability metadata",
            row=documents,
        )
    )
    entries.append(
        _corpus_entry(
            source="intelligence-store:signals",
            domain=SourceDomain.REGULATORY_EVENT,
            data_type="extracted event signals (all event types pooled)",
            row=signals,
        )
    )

    by_type = store.connection.execute(
        """
        SELECT json_extract_string(payload, '$.event_type') AS event_type,
               count(*), min(available_time), max(available_time)
        FROM signals GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    for event_type, count, first, last in by_type:
        entries.append(
            _corpus_entry(
                source=f"intelligence-store:signals[{event_type}]",
                domain=_domain_for_event_type(str(event_type)),
                data_type=f"extracted {event_type} events",
                row=(count, first, last),
            )
        )
    return entries


def _corpus_entry(
    *, source: str, domain: SourceDomain, data_type: str, row: Any
) -> SourceInventoryEntry:
    count = int(row[0]) if row and row[0] is not None else 0
    first = _as_utc(row[1]) if row and len(row) > 1 else None
    last = _as_utc(row[2]) if row and len(row) > 2 else None
    span_days = (last - first).days if first and last else 0
    return SourceInventoryEntry(
        source=source,
        domain=domain,
        data_type=data_type,
        start_date=first,
        end_date=last,
        frequency="event-driven" if count else "n/a",
        point_in_time_availability=(
            "available_at recorded at collection time" if count else "no observations collected"
        ),
        provenance="market_intelligence collection pipeline" if count else "no collection run persisted",
        license_status="depends on the configured provider contract",
        missingness="no coverage at all" if not count else f"{count} observations over {span_days} days",
        observation_count=count,
        availability=DataAvailability.AVAILABLE if count else DataAvailability.DATA_UNAVAILABLE,
        evidence_tier=EvidenceTier.PIT_VALIDATED if count else None,
        suitable_for_historical_study=count > 0,
        notes="" if count else "no historical corpus exists; event studies over this domain cannot be run",
    )


def _domain_for_event_type(event_type: str) -> SourceDomain:
    mapping = {
        "REGULATION": SourceDomain.REGULATORY_EVENT,
        "ETF_FLOW": SourceDomain.ETF_EVENT,
        "WHALE_TRANSFER": SourceDomain.ONCHAIN_WHALE,
        "ENTITY_STATEMENT": SourceDomain.ENTITY_STATEMENT,
        "MONETARY_POLICY": SourceDomain.MACRO_MARKET,
    }
    return mapping.get(event_type, SourceDomain.NEWS_WEB)


def _as_utc(value: Any) -> datetime | None:
    """DuckDB returns TIMESTAMPTZ as aware, so a naive value here means the
    column was written by something that bypassed the store. That is a defect
    worth surfacing, not one worth guessing a timezone for, so it raises."""
    if not isinstance(value, datetime):
        return None
    return require_utc(value, "stored timestamp")


def unavailable_entry(
    source: str,
    domain: SourceDomain,
    data_type: str,
    reason: str,
) -> SourceInventoryEntry:
    """An explicit DATA_UNAVAILABLE row, so absence is recorded, not implied."""
    return SourceInventoryEntry(
        source=source,
        domain=domain,
        data_type=data_type,
        start_date=None,
        end_date=None,
        frequency="n/a",
        point_in_time_availability="n/a",
        provenance="none",
        license_status="n/a",
        missingness="complete",
        observation_count=0,
        availability=DataAvailability.DATA_UNAVAILABLE,
        evidence_tier=None,
        suitable_for_historical_study=False,
        notes=reason,
    )


# ---------------------------------------------------------------- coverage bias


class CoverageBiasResult(BaseModel):
    """B4.34. Do well-covered periods differ from poorly-covered ones?"""

    model_config = ConfigDict(frozen=True)

    high_coverage_origins: int
    low_coverage_origins: int
    high_coverage_mean_abs_return: float | None
    low_coverage_mean_abs_return: float | None
    absolute_difference: float | None
    high_coverage_span: tuple[datetime, datetime] | None
    low_coverage_span: tuple[datetime, datetime] | None
    verdict: str


def coverage_bias(
    rows: Sequence[dict[str, Any]],
    *,
    coverage_field: str = "provider_coverage_ratio",
    outcome_field: str,
    threshold: float = 1.0,
) -> CoverageBiasResult:
    """Compare outcome dispersion in high- and low-coverage periods.

    A signal that only appears when intelligence coverage was good may be
    telling you about the collector's uptime rather than about the market. The
    check is deliberately blunt — a split and two means — because anything more
    elaborate invites tuning.
    """
    high: list[float] = []
    low: list[float] = []
    high_times: list[datetime] = []
    low_times: list[datetime] = []

    for row in rows:
        coverage = row.get(coverage_field)
        outcome = row.get(outcome_field)
        origin = row.get("forecast_origin")
        if coverage is None or outcome is None or not isinstance(origin, datetime):
            continue
        bucket_values, bucket_times = (high, high_times) if float(coverage) >= threshold else (low, low_times)
        bucket_values.append(abs(float(outcome)))
        bucket_times.append(origin)

    high_mean = sum(high) / len(high) if high else None
    low_mean = sum(low) / len(low) if low else None
    difference = abs(high_mean - low_mean) if high_mean is not None and low_mean is not None else None

    if not high or not low:
        verdict = "INSUFFICIENT — one coverage bucket is empty, so no comparison is possible"
    elif difference is not None and low_mean and difference / max(low_mean, 1e-12) > 0.25:
        verdict = "COVERAGE BIAS PRESENT — outcome dispersion differs by more than 25% between buckets"
    else:
        verdict = "NO MATERIAL COVERAGE BIAS DETECTED at this threshold"

    return CoverageBiasResult(
        high_coverage_origins=len(high),
        low_coverage_origins=len(low),
        high_coverage_mean_abs_return=high_mean,
        low_coverage_mean_abs_return=low_mean,
        absolute_difference=difference,
        high_coverage_span=(min(high_times), max(high_times)) if high_times else None,
        low_coverage_span=(min(low_times), max(low_times)) if low_times else None,
        verdict=verdict,
    )


def write_inventory(entries: Sequence[SourceInventoryEntry], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.as_record() for entry in entries]
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


__all__ = [
    "CoverageBiasResult",
    "SourceInventoryEntry",
    "coverage_bias",
    "inventory_intelligence_store",
    "inventory_market_series",
    "unavailable_entry",
    "write_inventory",
]
