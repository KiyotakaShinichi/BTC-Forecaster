"""B4.4 – B4.14 — one reusable event-study engine.

A spec, not a notebook. Every entity, event type and threshold is a field on
`EventStudySpec`, so the same code path produces the Trump study, the whale
study and the placebo study, and none of them can quietly diverge in how they
filter, cluster or window.

Three parts of this are where event studies usually go wrong.

**Forty articles are not forty events (B4.9).** One real-world announcement
produces a wave of coverage; counting each story as an independent observation
inflates N by an order of magnitude and shrinks every confidence interval to
match. Events are clustered by (type, entity) within a declared window, and the
number of source documents is kept as corroboration metadata rather than as
sample size.

**Overlapping windows are not independent (B4.10).** Two events six hours apart
share most of a 72-hour outcome window. The policy is declared per study rather
than assumed, and the effective event count -- what the study is really powered
on -- is reported next to the raw count.

**A move that already happened is not a response (B4.11).** Pre-event returns are
computed for every study. If BTC had already moved before the event, the honest
readings are reverse causation or leakage, not impact, and the result carries
that caveat rather than a causal claim.

Nothing here decides anything. It computes described quantities and hands them
to `stats` for inference and to `registry` for a decision.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .contracts import B4DataError, EvidenceTier, MarketBar, MarketSeries, require_utc
from .market_data import close_as_of, last_available_index
from .stats import BootstrapConfig, BootstrapResult, DescriptiveStats, block_bootstrap, describe
from .targets import horizon_delta


class OverlapPolicy(str, Enum):
    """B4.10. How to treat events whose outcome windows overlap."""

    #: Keep the first event of each cluster and drop the rest. Simplest to
    #: explain; discards real repeat information.
    KEEP_FIRST = "KEEP_FIRST"
    #: One observation per cluster, outcomes averaged. Keeps every event's
    #: contribution while giving the cluster a single vote.
    AGGREGATE_CLUSTER = "AGGREGATE_CLUSTER"
    #: Keep every event and let the block bootstrap carry the dependence.
    #: Only defensible when the block length covers the overlap.
    ALL_DEPENDENCE_AWARE = "ALL_DEPENDENCE_AWARE"


class BenchmarkDefinition(str, Enum):
    """B4.14. What the response is measured against."""

    RAW = "RAW"
    #: BTC return minus an equal-weighted return of other large-cap crypto.
    #: Removes market-wide moves without importing any Track A model.
    CRYPTO_MARKET_EXCESS = "CRYPTO_MARKET_EXCESS"


class StudyEvent(BaseModel):
    """One historical event, normalised for study.

    `available_at` is separate from `event_time` throughout: an event that
    happened at 02:00 but only became public at 09:00 must be studied from
    09:00, and conflating the two is a leak that looks like a strong result.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: str
    entity: str
    event_time: datetime
    available_at: datetime
    btc_relevance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    sentiment: float = Field(ge=-1.0, le=1.0)
    transfer_context: str | None = None
    source_ids: tuple[str, ...] = ()
    provider: str = "unknown"
    evidence_tier: EvidenceTier = EvidenceTier.PIT_VALIDATED

    def normalised(self) -> "StudyEvent":
        return self.model_copy(
            update={
                "event_time": require_utc(self.event_time, "event_time"),
                "available_at": require_utc(self.available_at, "available_at"),
            }
        )


class EventFilter(BaseModel):
    """B4.4 / B4.32. Declared before results; never tuned on returns."""

    model_config = ConfigDict(frozen=True)

    event_types: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    transfer_contexts: tuple[str, ...] = ()
    minimum_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    require_evidence_tier: EvidenceTier = EvidenceTier.PIT_VALIDATED
    require_provenance: bool = True


class EventStudySpec(BaseModel):
    """B4.4. The whole definition of one study, hashable and frozen."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    event_filter: EventFilter
    #: Negative offsets, in hours, for the pre-event trend (B4.11).
    pre_window_hours: tuple[int, ...] = (-24, -6, -1)
    post_horizons: tuple[str, ...] = ("1h", "6h", "24h", "72h")
    benchmark: BenchmarkDefinition = BenchmarkDefinition.RAW
    minimum_event_count: int = 30
    bootstrap: BootstrapConfig = BootstrapConfig()
    multiple_testing_family: str = "default"
    overlap_policy: OverlapPolicy = OverlapPolicy.AGGREGATE_CLUSTER
    #: Events of the same (type, entity) closer together than this belong to one
    #: cluster. B4.9's "forty articles, one event".
    cluster_window_hours: int = Field(default=6, ge=0)

    def spec_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


class EventCluster(BaseModel):
    """A group of events judged to describe one underlying occurrence."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    events: tuple[StudyEvent, ...]

    @property
    def anchor(self) -> StudyEvent:
        """Earliest availability: when the market could first have known."""
        return self.events[0]

    @property
    def source_count(self) -> int:
        """B4.9. Corroboration breadth, kept as metadata, never as sample size."""
        unique: set[str] = set()
        for event in self.events:
            unique.update(event.source_ids or (event.event_id,))
        return len(unique)

    @property
    def provider_count(self) -> int:
        return len({event.provider for event in self.events})

    @property
    def entity_count(self) -> int:
        return len({event.entity for event in self.events})


class EventObservation(BaseModel):
    """One row a study is actually powered on."""

    model_config = ConfigDict(frozen=True)

    observation_id: str
    origin: datetime
    event_type: str
    entity: str
    provider: str
    member_event_count: int
    source_count: int
    pre_event_returns: dict[str, float | None]
    responses: dict[str, float | None]
    abs_responses: dict[str, float | None]


class HorizonResult(BaseModel):
    """B4.13. Everything about one horizon, before any single verdict."""

    model_config = ConfigDict(frozen=True)

    horizon: str
    descriptive: DescriptiveStats
    bootstrap: BootstrapResult | None
    abs_descriptive: DescriptiveStats


class EventStudyResult(BaseModel):
    """The full output of one study. Deliberately not reduced to a p-value."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    spec_hash: str
    benchmark: BenchmarkDefinition
    overlap_policy: OverlapPolicy
    raw_event_count: int
    cluster_count: int
    effective_event_count: int
    observation_count: int
    origin_span: tuple[datetime, datetime] | None
    pre_event_trend: dict[str, DescriptiveStats]
    horizons: dict[str, HorizonResult]
    insufficient: bool
    notes: tuple[str, ...] = ()


# ------------------------------------------------------------------ filtering


def filter_events(events: Sequence[StudyEvent], spec: EventStudySpec) -> list[StudyEvent]:
    """Apply the declared filter. Order is stable and by availability."""
    criteria = spec.event_filter
    kept: list[StudyEvent] = []
    for raw in events:
        event = raw.normalised()
        if criteria.event_types and event.event_type not in criteria.event_types:
            continue
        if criteria.entities and event.entity not in criteria.entities:
            continue
        if criteria.transfer_contexts and (event.transfer_context or "UNKNOWN") not in criteria.transfer_contexts:
            continue
        if event.btc_relevance < criteria.minimum_relevance:
            continue
        if event.confidence < criteria.minimum_confidence:
            continue
        if event.novelty < criteria.minimum_novelty:
            continue
        if event.evidence_tier is not criteria.require_evidence_tier:
            continue
        if criteria.require_provenance and not event.source_ids:
            continue
        kept.append(event)
    return sorted(kept, key=lambda event: (event.available_at, event.event_id))


# ----------------------------------------------------------------- clustering


def cluster_events(events: Sequence[StudyEvent], spec: EventStudySpec) -> list[EventCluster]:
    """B4.9. Greedy, deterministic clustering by (type, entity) within a window.

    Greedy chaining is chosen over a similarity model because it is explicable
    and reproducible: an event joins the open cluster for its (type, entity) if
    it becomes available within the window of the *previous* member. Chaining
    means a genuine multi-day news cycle stays one cluster rather than splitting
    at an arbitrary cutoff, which is the behaviour the inflation problem needs.
    """
    window = timedelta(hours=spec.cluster_window_hours)
    open_clusters: dict[tuple[str, str], list[StudyEvent]] = {}
    finished: list[list[StudyEvent]] = []

    for event in sorted(events, key=lambda item: (item.available_at, item.event_id)):
        key = (event.event_type, event.entity)
        current = open_clusters.get(key)
        if current is not None and event.available_at - current[-1].available_at <= window:
            current.append(event)
            continue
        if current is not None:
            finished.append(current)
        open_clusters[key] = [event]

    finished.extend(open_clusters.values())
    finished.sort(key=lambda members: (members[0].available_at, members[0].event_id))
    return [
        EventCluster(
            cluster_id=hashlib.sha256(
                "|".join(member.event_id for member in members).encode()
            ).hexdigest()[:16],
            events=tuple(members),
        )
        for members in finished
    ]


def effective_event_count(observations: Sequence[EventObservation], horizon: str) -> int:
    """B4.10. How many non-overlapping windows the sample really contains.

    A greedy sweep: walk the observations in time and count one whenever the
    next origin lands outside the previous counted window. It is a lower bound
    on independence, and it is reported alongside the raw count so a reader can
    see when 400 events are really worth 40.
    """
    span = horizon_delta(horizon)
    count = 0
    frontier: datetime | None = None
    for observation in sorted(observations, key=lambda item: item.origin):
        if frontier is None or observation.origin >= frontier:
            count += 1
            frontier = observation.origin + span
    return count


# ------------------------------------------------------------------- response


def _return_between(series: MarketSeries, start: datetime, end: datetime) -> float | None:
    """Return between the closes available at two instants, or None."""
    start_index = last_available_index(series, start)
    end_index = last_available_index(series, end)
    if start_index < 0 or end_index < 0 or end_index == start_index:
        return None
    if series.end is not None and end > series.end:
        return None
    return series.bars[end_index].close / series.bars[start_index].close - 1.0


def pre_event_returns(
    series: MarketSeries, origin: datetime, offsets_hours: Sequence[int]
) -> dict[str, float | None]:
    """B4.11. Returns from each pre-window offset up to the origin."""
    result: dict[str, float | None] = {}
    for offset in offsets_hours:
        if offset >= 0:
            raise B4DataError(f"pre-window offsets must be negative, got {offset}")
        label = f"{offset}h"
        result[label] = _return_between(series, origin + timedelta(hours=offset), origin)
    return result


def event_response(
    series: MarketSeries,
    origin: datetime,
    horizons: Sequence[str],
    benchmark_series: MarketSeries | None = None,
) -> dict[str, float | None]:
    """Forward response at each declared horizon, optionally benchmark-adjusted.

    Benchmark adjustment is a plain difference of contemporaneous returns. A
    fitted beta would be a model, and B4.14 asks for a simple external-signal
    comparison rather than an imported one.
    """
    responses: dict[str, float | None] = {}
    for horizon in horizons:
        end = origin + horizon_delta(horizon)
        raw = _return_between(series, origin, end)
        if raw is None:
            responses[horizon] = None
            continue
        if benchmark_series is None:
            responses[horizon] = raw
            continue
        benchmark = _return_between(benchmark_series, origin, end)
        responses[horizon] = None if benchmark is None else raw - benchmark
    return responses


def equal_weighted_benchmark(series_list: Sequence[MarketSeries], series_id: str = "crypto_ex_btc") -> MarketSeries:
    """B4.14. Equal-weighted index of the constituents, on their common grid.

    Built as a synthetic price level so the same `_return_between` machinery
    applies to it. Only timestamps present in *every* constituent are used: a
    benchmark whose composition changes between two dates would attribute the
    composition change to the market.
    """
    if not series_list:
        raise B4DataError("a benchmark needs at least one constituent")
    periods = {series.period_seconds for series in series_list}
    if len(periods) != 1:
        raise B4DataError(f"benchmark constituents have mixed periods {sorted(periods)}")

    common: set[datetime] | None = None
    for series in series_list:
        stamps = {bar.period_start for bar in series.bars}
        common = stamps if common is None else (common & stamps)
    ordered = sorted(common or set())
    if len(ordered) < 2:
        raise B4DataError("benchmark constituents share fewer than two common periods")

    lookup = [{bar.period_start: bar.close for bar in series.bars} for series in series_list]
    level = 100.0
    bars = []
    bars.append(
        MarketBar(
            period_start=ordered[0],
            period_seconds=series_list[0].period_seconds,
            open=level,
            high=level,
            low=level,
            close=level,
            volume=0.0,
        )
    )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        growth = sum(table[current] / table[previous] for table in lookup) / len(lookup)
        level *= growth
        bars.append(
            MarketBar(
                period_start=current,
                period_seconds=series_list[0].period_seconds,
                open=level,
                high=level,
                low=level,
                close=level,
                volume=0.0,
            )
        )

    return MarketSeries(
        series_id=series_id,
        ticker="+".join(series.ticker for series in series_list),
        domain=series_list[0].domain,
        provider="derived",
        timestamp_convention=series_list[0].timestamp_convention,
        evidence_tier=series_list[0].evidence_tier,
        source_timezone="UTC",
        bars=tuple(bars),
    )


# ------------------------------------------------------------------ the study


def build_observations(
    clusters: Sequence[EventCluster],
    spec: EventStudySpec,
    series: MarketSeries,
    benchmark_series: MarketSeries | None = None,
) -> list[EventObservation]:
    """Turn clusters into study rows under the declared overlap policy."""
    observations: list[EventObservation] = []

    for cluster in clusters:
        if spec.overlap_policy is OverlapPolicy.ALL_DEPENDENCE_AWARE:
            members: Sequence[StudyEvent] = cluster.events
        else:
            members = (cluster.anchor,)

        if spec.overlap_policy is OverlapPolicy.AGGREGATE_CLUSTER and len(cluster.events) > 1:
            # One vote per cluster, but every member's own response contributes
            # to it. KEEP_FIRST would throw the later members away; averaging
            # keeps their information without letting a 40-article news cycle
            # become 40 observations.
            observation = _aggregate_cluster(cluster, spec, series, benchmark_series)
            if observation is not None:
                observations.append(observation)
            continue

        for member in members:
            origin = member.available_at
            if close_as_of(series, origin) is None:
                continue  # the market series does not cover this event
            responses = event_response(series, origin, spec.post_horizons, benchmark_series)
            observations.append(
                EventObservation(
                    observation_id=f"{cluster.cluster_id}:{member.event_id}",
                    origin=origin,
                    event_type=member.event_type,
                    entity=member.entity,
                    provider=member.provider,
                    member_event_count=len(cluster.events),
                    source_count=cluster.source_count,
                    pre_event_returns=pre_event_returns(series, origin, spec.pre_window_hours),
                    responses=responses,
                    abs_responses={
                        horizon: (None if value is None else abs(value)) for horizon, value in responses.items()
                    },
                )
            )

    return sorted(observations, key=lambda item: (item.origin, item.observation_id))


def _aggregate_cluster(
    cluster: EventCluster,
    spec: EventStudySpec,
    series: MarketSeries,
    benchmark_series: MarketSeries | None,
) -> EventObservation | None:
    """Mean response across a cluster's members, as a single observation."""
    per_member: list[tuple[datetime, dict[str, float | None], dict[str, float | None]]] = []
    for member in cluster.events:
        origin = member.available_at
        if close_as_of(series, origin) is None:
            continue
        per_member.append(
            (
                origin,
                event_response(series, origin, spec.post_horizons, benchmark_series),
                pre_event_returns(series, origin, spec.pre_window_hours),
            )
        )
    if not per_member:
        return None

    def mean_over_members(keys: Sequence[str], *, post: bool) -> dict[str, float | None]:
        merged: dict[str, float | None] = {}
        for key in keys:
            values = [
                value
                for _, responses, pre in per_member
                if (value := (responses if post else pre).get(key)) is not None
            ]
            merged[key] = sum(values) / len(values) if values else None
        return merged

    responses = mean_over_members(spec.post_horizons, post=True)
    pre_returns = mean_over_members(tuple(f"{offset}h" for offset in spec.pre_window_hours), post=False)
    anchor_origin = per_member[0][0]
    return EventObservation(
        observation_id=f"{cluster.cluster_id}:aggregate",
        origin=anchor_origin,
        event_type=cluster.anchor.event_type,
        entity=cluster.anchor.entity,
        provider=cluster.anchor.provider,
        member_event_count=len(cluster.events),
        source_count=cluster.source_count,
        pre_event_returns=pre_returns,
        responses=responses,
        abs_responses={
            horizon: (None if value is None else abs(value)) for horizon, value in responses.items()
        },
    )


def run_event_study(
    events: Sequence[StudyEvent],
    spec: EventStudySpec,
    series: MarketSeries,
    benchmark_series: MarketSeries | None = None,
) -> EventStudyResult:
    """B4.4 – B4.14 end to end, for one declared study.

    Returns a result even when there is nothing to study. An empty result with
    `insufficient=True` and an explanatory note is the honest output for a
    domain with no data, and it keeps the absence in the results table instead
    of leaving a gap a reader has to interpret.
    """
    eligible = filter_events(events, spec)
    clusters = cluster_events(eligible, spec)
    observations = build_observations(clusters, spec, series, benchmark_series)

    notes: list[str] = []
    if spec.benchmark is BenchmarkDefinition.CRYPTO_MARKET_EXCESS and benchmark_series is None:
        raise B4DataError(f"{spec.study_id}: CRYPTO_MARKET_EXCESS requires a benchmark series")
    if spec.benchmark is BenchmarkDefinition.RAW and benchmark_series is not None:
        raise B4DataError(f"{spec.study_id}: a RAW study must not be handed a benchmark series")

    effective = min(
        (effective_event_count(observations, horizon) for horizon in spec.post_horizons),
        default=0,
    )
    insufficient = len(observations) < spec.minimum_event_count
    if insufficient:
        notes.append(
            f"{len(observations)} observations is below the declared minimum of "
            f"{spec.minimum_event_count}; no inference is reported"
        )
    if observations and effective < spec.minimum_event_count:
        notes.append(
            f"effective (non-overlapping) count at the longest horizon is {effective}, "
            f"below the declared minimum of {spec.minimum_event_count}"
        )

    pre_trend: dict[str, DescriptiveStats] = {}
    for offset in spec.pre_window_hours:
        label = f"{offset}h"
        pre_trend[label] = describe(
            [
                value
                for observation in observations
                if (value := observation.pre_event_returns.get(label)) is not None
            ]
        )

    horizons: dict[str, HorizonResult] = {}
    for horizon in spec.post_horizons:
        values = [
            value for observation in observations if (value := observation.responses.get(horizon)) is not None
        ]
        abs_values = [
            value
            for observation in observations
            if (value := observation.abs_responses.get(horizon)) is not None
        ]
        bootstrap: BootstrapResult | None = None
        if not insufficient and len(values) >= 2:
            bootstrap = block_bootstrap(values, spec.bootstrap)
        horizons[horizon] = HorizonResult(
            horizon=horizon,
            descriptive=describe(values),
            bootstrap=bootstrap,
            abs_descriptive=describe(abs_values),
        )

    span = (
        (observations[0].origin, observations[-1].origin)
        if observations
        else None
    )

    return EventStudyResult(
        study_id=spec.study_id,
        spec_hash=spec.spec_hash(),
        benchmark=spec.benchmark,
        overlap_policy=spec.overlap_policy,
        raw_event_count=len(eligible),
        cluster_count=len(clusters),
        effective_event_count=effective,
        observation_count=len(observations),
        origin_span=span,
        pre_event_trend=pre_trend,
        horizons=horizons,
        insufficient=insufficient,
        notes=tuple(notes),
    )


def pre_trend_warning(result: EventStudyResult, horizon: str, ratio: float = 0.5) -> str | None:
    """B4.11. Flag a study whose move mostly preceded its events.

    Compares the mean pre-event move against the mean response. When the
    pre-move is a large fraction of the response, the reading is reverse timing
    or leakage rather than impact, and the caller must say so rather than
    describing a causal effect.
    """
    horizon_result = result.horizons.get(horizon)
    if horizon_result is None or horizon_result.descriptive.mean is None:
        return None
    response = abs(horizon_result.descriptive.mean)
    if response == 0.0:
        return None
    for label, stats in result.pre_event_trend.items():
        if stats.mean is None:
            continue
        if abs(stats.mean) >= ratio * response:
            return (
                f"pre-event mean over {label} is {stats.mean:+.5f}, at least {ratio:g}x the "
                f"{horizon} response of {horizon_result.descriptive.mean:+.5f}: consistent with "
                "reverse timing or leakage, not with event impact"
            )
    return None


__all__ = [
    "BenchmarkDefinition",
    "EventCluster",
    "EventFilter",
    "EventObservation",
    "EventStudyResult",
    "EventStudySpec",
    "HorizonResult",
    "OverlapPolicy",
    "StudyEvent",
    "build_observations",
    "cluster_events",
    "effective_event_count",
    "equal_weighted_benchmark",
    "event_response",
    "filter_events",
    "pre_event_returns",
    "pre_trend_warning",
    "run_event_study",
]
