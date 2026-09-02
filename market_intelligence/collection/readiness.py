"""B4.1.30 – B4.1.33 — when is there enough evidence to re-run B4?

Not "in six months". A calendar date is a guess about collection rates dressed
as a criterion, and it would authorise an underpowered study on the day it
arrives regardless of what was actually collected.

Instead the gate is the same arithmetic B4 already froze: minimum observations,
minimum *effective* non-overlapping observations, source diversity, temporal
span. Those numbers come from B4's carry-forward policy rather than being
invented here, so readiness cannot be met by writing a laxer threshold — the
thresholds already exist, in a preregistration that is hashed.

The one thing this module must never do is help. It is tempting, on finding a
family two events short, to widen a window or pool a category. B4.1.32 is
explicit that whale contexts are never merged to hit a minimum, and the same
applies everywhere: `NOT_READY` is a finding, and the correct response is to
keep collecting.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..models import TransferContext
from .clustering import EventCluster, effective_non_overlapping


class Readiness(str, Enum):
    """B4.1.30. The three states, and nothing in between."""

    NOT_READY = "NOT_READY"
    PARTIALLY_READY = "PARTIALLY_READY"
    READY_FOR_VALIDATION = "READY_FOR_VALIDATION"


class AdequacyPolicy(BaseModel):
    """Thresholds, carried from B4 rather than chosen here."""

    model_config = ConfigDict(frozen=True)

    #: B4's preregistered minimum event count.
    minimum_events: int = Field(default=30, ge=1)
    #: B4's preregistered minimum effective (non-overlapping) count.
    minimum_effective_events: int = Field(default=20, ge=1)
    #: An effect estimated from one publisher is about that publisher.
    minimum_publishers: int = Field(default=3, ge=1)
    minimum_providers: int = Field(default=1, ge=1)
    #: A study spanning one month cannot separate an effect from a month.
    minimum_span_days: int = Field(default=180, ge=1)
    #: Horizon the effective count is computed at. The longest horizon B4 tests,
    #: because that is where overlap bites hardest.
    horizon_hours: int = Field(default=168, ge=1)
    #: Fraction of days in the span that must have at least one collection run.
    minimum_coverage_fraction: float = Field(default=0.8, ge=0.0, le=1.0)


#: The default policy, built once. B4's own frozen thresholds; a caller that
#: wants different numbers passes them explicitly rather than editing these.
DEFAULT_POLICY = AdequacyPolicy()


class FamilyReadiness(BaseModel):
    """One study family's standing against the policy."""

    model_config = ConfigDict(frozen=True)

    family: str
    events: int
    effective_events: int
    publishers: int
    providers: int
    span_days: int
    coverage_fraction: float
    readiness: Readiness
    #: Every clause that is not yet met, so "not ready" is actionable rather
    #: than a verdict to argue with.
    unmet: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.readiness is Readiness.READY_FOR_VALIDATION


def assess_family(
    family: str,
    clusters: Sequence[EventCluster],
    *,
    policy: AdequacyPolicy = DEFAULT_POLICY,
    coverage_fraction: float = 0.0,
) -> FamilyReadiness:
    """Judge one family. Reports every unmet clause, not just the first."""
    events = len(clusters)
    effective = effective_non_overlapping(clusters, policy.horizon_hours)
    # Source diversity is a property of the *family*, not of its best-covered
    # event. Taking `max(cluster.publisher_count)` measured per-event
    # corroboration instead, and under continuous collection of official feeds
    # each announcement has exactly one primary publisher -- so that number
    # stays at 1 forever and the publisher clause could never be satisfied no
    # matter how long collection ran. The union is what the threshold means.
    publishers = len({publisher for cluster in clusters for publisher in cluster.publishers})
    providers = len({provider for cluster in clusters for provider in cluster.providers})

    span_days = 0
    if clusters:
        ordered = sorted(clusters, key=lambda item: item.first_available_at)
        span_days = (ordered[-1].first_available_at - ordered[0].first_available_at).days

    unmet: list[str] = []
    if events < policy.minimum_events:
        unmet.append(f"{events} events, need {policy.minimum_events}")
    if effective < policy.minimum_effective_events:
        unmet.append(
            f"{effective} effective (non-overlapping at {policy.horizon_hours}h), "
            f"need {policy.minimum_effective_events}"
        )
    if publishers < policy.minimum_publishers:
        unmet.append(f"{publishers} publishers, need {policy.minimum_publishers}")
    if providers < policy.minimum_providers:
        unmet.append(f"{providers} providers, need {policy.minimum_providers}")
    if span_days < policy.minimum_span_days:
        unmet.append(f"{span_days} day span, need {policy.minimum_span_days}")
    if coverage_fraction < policy.minimum_coverage_fraction:
        unmet.append(
            f"collection covered {coverage_fraction:.0%} of days, need "
            f"{policy.minimum_coverage_fraction:.0%}"
        )

    if not unmet:
        readiness = Readiness.READY_FOR_VALIDATION
    elif events == 0:
        readiness = Readiness.NOT_READY
    elif len(unmet) <= 2 and events >= policy.minimum_events // 2:
        # Something real is accumulating and more than half the bar is met.
        # Deliberately still not READY: partial is a progress report, never a
        # licence to run the study.
        readiness = Readiness.PARTIALLY_READY
    else:
        readiness = Readiness.NOT_READY

    return FamilyReadiness(
        family=family,
        events=events,
        effective_events=effective,
        publishers=publishers,
        providers=providers,
        span_days=span_days,
        coverage_fraction=coverage_fraction,
        readiness=readiness,
        unmet=tuple(unmet),
    )


def assess_entities(
    clusters: Sequence[EventCluster],
    entities: Sequence[str],
    *,
    policy: AdequacyPolicy = DEFAULT_POLICY,
    coverage_fraction: float = 0.0,
) -> list[FamilyReadiness]:
    """B4.1.31. Every named entity, including the ones with nothing.

    An entity with no evidence still gets a row. A missing row would read as
    "not asked about", and B4's whole failure mode was absence being mistaken
    for something else.
    """
    return [
        assess_family(
            f"entity:{entity}",
            [cluster for cluster in clusters if (cluster.entity or "") == entity],
            policy=policy,
            coverage_fraction=coverage_fraction,
        )
        for entity in entities
    ]


def assess_whale_contexts(
    clusters_by_context: Mapping[str, Sequence[EventCluster]],
    *,
    policy: AdequacyPolicy = DEFAULT_POLICY,
    coverage_fraction: float = 0.0,
) -> list[FamilyReadiness]:
    """B4.1.32. Each transfer context separately. Never pooled to reach N."""
    return [
        assess_family(
            f"whale:{context.value}",
            clusters_by_context.get(context.value, ()),
            policy=policy,
            coverage_fraction=coverage_fraction,
        )
        for context in TransferContext
    ]


def assess_sentiment(
    clusters: Sequence[EventCluster],
    composites: Sequence[str],
    *,
    policy: AdequacyPolicy = DEFAULT_POLICY,
    coverage_fraction: float = 0.0,
) -> list[FamilyReadiness]:
    """B4.1.33. Whether the predefined composites have evidence to be built on.

    Every composite is judged against the same event pool, because they are
    three transformations of one set of extracted scores rather than three
    independent samples. Thresholds are not tuned against BTC returns -- nothing
    in this module has ever seen a return.
    """
    return [
        assess_family(f"sentiment:{composite}", clusters, policy=policy, coverage_fraction=coverage_fraction)
        for composite in composites
    ]


def overall_readiness(families: Sequence[FamilyReadiness]) -> Readiness:
    """The corpus is only as ready as its families.

    READY requires every family ready. One ready family among twenty means a
    single study could run, not that B4 can be re-run -- and B4 is the thing
    this gate authorises.
    """
    if not families:
        return Readiness.NOT_READY
    if all(family.ready for family in families):
        return Readiness.READY_FOR_VALIDATION
    if any(family.readiness is not Readiness.NOT_READY for family in families):
        return Readiness.PARTIALLY_READY
    return Readiness.NOT_READY


def coverage_fraction(collection_days: Sequence[datetime], span_start: datetime, span_end: datetime) -> float:
    """B4.1.23. Days with at least one successful run, over days in the span.

    Distinct calendar days, not run count: ten runs on one day is one day of
    coverage, and counting runs would let a busy afternoon disguise a silent week.
    """
    if span_end <= span_start:
        return 0.0
    total_days = max(1, (span_end - span_start).days + 1)
    covered = {moment.date() for moment in collection_days}
    return min(1.0, len(covered) / total_days)


def days_until_ready(family: FamilyReadiness, events_per_day: float, policy: AdequacyPolicy) -> int | None:
    """A projection, explicitly labelled as one.

    Linear extrapolation from the observed rate. It says nothing about whether
    the rate will hold, and it is reported so an operator can plan rather than
    guess -- never as a readiness criterion in its own right.
    """
    if family.ready:
        return 0
    if events_per_day <= 0.0:
        return None
    shortfall = max(0, policy.minimum_events - family.events)
    span_shortfall = max(0, policy.minimum_span_days - family.span_days)
    return max(int(shortfall / events_per_day + 0.999), span_shortfall) or None


def span_of(clusters: Sequence[EventCluster]) -> tuple[datetime, datetime] | None:
    if not clusters:
        return None
    ordered = sorted(clusters, key=lambda item: item.first_available_at)
    return ordered[0].first_available_at, ordered[-1].last_available_at


def projected_span_days(clusters: Sequence[EventCluster]) -> int:
    span = span_of(clusters)
    return (span[1] - span[0]).days if span else 0


DEFAULT_HORIZON = timedelta(hours=DEFAULT_POLICY.horizon_hours)

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_POLICY",
    "AdequacyPolicy",
    "FamilyReadiness",
    "Readiness",
    "assess_entities",
    "assess_family",
    "assess_sentiment",
    "assess_whale_contexts",
    "coverage_fraction",
    "days_until_ready",
    "overall_readiness",
    "projected_span_days",
    "span_of",
]
