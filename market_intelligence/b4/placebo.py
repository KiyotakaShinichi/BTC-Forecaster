"""B4.18 / B4.19 — placebo tests and matched non-event controls.

A confidence interval answers "how uncertain is this estimate". It does not
answer the question that actually matters here: *would an effect this large turn
up anyway, from a procedure this elaborate, applied to events that mean nothing?*

Three ways of asking it, all preserving temporal structure — a placebo that
destroys the dependence structure is testing a different, easier null and will
happily clear a spurious result:

**Shifted timestamps.** Each event moves by a random offset inside a block. The
count, the clustering and the calendar footprint survive; only the alignment to
the market is broken.

**Permuted labels.** Event labels are shuffled *within* time blocks, so a study
of one entity gets another entity's dates from the same era. This is the sharper
test for entity studies, because it holds the overall event process fixed.

**Matched non-event origins.** Every event origin is paired with a non-event
origin from the same calendar period and volatility bucket. If event origins
behave like their matches, the event is not adding information.

Matching stays deliberately crude — a coarse volatility bucket and a calendar
period, nothing fitted. This is observational; an elaborate propensity model
would add the *appearance* of causal identification without the design that
would earn it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from .contracts import B4DataError, MarketSeries
from .eventstudy import (
    EventStudySpec,
    StudyEvent,
    build_observations,
    cluster_events,
    event_response,
    filter_events,
)
from .market_data import close_as_of, last_available_index
from .stats import DescriptiveStats, describe

DEFAULT_PLACEBO_REPLICATES = 500


class PlaceboResult(BaseModel):
    """B4.18. Where the observed effect sits in the placebo distribution."""

    model_config = ConfigDict(frozen=True)

    horizon: str
    method: str
    replicates: int
    seed: int
    observed_effect: float
    placebo_mean: float
    placebo_std: float
    #: Fraction of placebo replicates whose |effect| is at least the observed
    #: one. This is the number that matters: a small value means the procedure
    #: does not routinely manufacture effects this large from nothing.
    exceedance_rate: float
    placebo_quantiles: dict[str, float]
    verdict: str


class MatchedControlResult(BaseModel):
    """B4.19. Event origins against comparable non-event origins."""

    model_config = ConfigDict(frozen=True)

    horizon: str
    matched_pairs: int
    unmatched_events: int
    event_stats: DescriptiveStats
    control_stats: DescriptiveStats
    mean_difference: float | None
    #: Standard error of the difference of means. The verdict is stated in
    #: multiples of this rather than of a single-observation sd: with 22 pairs
    #: the noise in a mean difference is roughly 0.3 sd, so a fixed "quarter of
    #: an sd" rule would call almost every null comparison a difference.
    difference_standard_error: float | None
    verdict: str


@dataclass(frozen=True)
class _Match:
    event_origin: datetime
    control_origin: datetime


def shift_events_within_blocks(
    events: Sequence[StudyEvent],
    *,
    block_hours: int,
    generator: random.Random,
) -> list[StudyEvent]:
    """Move each event by a random offset inside its block.

    The block bounds the displacement so an event stays in its own market
    regime: shifting a 2021 event into 2023 would test against a completely
    different volatility environment and make the placebo trivially easy to beat.
    """
    if block_hours <= 0:
        raise B4DataError("placebo block length must be positive")
    shifted: list[StudyEvent] = []
    for event in events:
        offset = timedelta(hours=generator.uniform(-block_hours, block_hours))
        shifted.append(
            event.model_copy(
                update={
                    "event_time": event.event_time + offset,
                    "available_at": event.available_at + offset,
                }
            )
        )
    return sorted(shifted, key=lambda item: (item.available_at, item.event_id))


def permute_labels_within_blocks(
    events: Sequence[StudyEvent],
    *,
    block_hours: int,
    generator: random.Random,
) -> list[StudyEvent]:
    """Shuffle (event_type, entity) labels among events in the same block.

    Timestamps and the overall event process are untouched, so the placebo asks
    exactly one question: does *this* label carry information the others do not?
    """
    if not events:
        return []
    ordered = sorted(events, key=lambda item: (item.available_at, item.event_id))
    span = timedelta(hours=block_hours)
    start = ordered[0].available_at

    blocks: dict[int, list[int]] = {}
    for index, event in enumerate(ordered):
        key = int((event.available_at - start) / span)
        blocks.setdefault(key, []).append(index)

    permuted = list(ordered)
    for indices in blocks.values():
        labels = [(ordered[index].event_type, ordered[index].entity) for index in indices]
        generator.shuffle(labels)
        for index, (event_type, entity) in zip(indices, labels, strict=True):
            permuted[index] = ordered[index].model_copy(
                update={"event_type": event_type, "entity": entity}
            )
    return permuted


def placebo_test(
    events: Sequence[StudyEvent],
    spec: EventStudySpec,
    series: MarketSeries,
    horizon: str,
    *,
    method: str = "shift",
    replicates: int = DEFAULT_PLACEBO_REPLICATES,
    seed: int = 20260831,
    block_hours: int = 24 * 30,
    benchmark_series: MarketSeries | None = None,
) -> PlaceboResult:
    """Compare the observed mean response against a placebo distribution."""
    if method not in {"shift", "permute"}:
        raise B4DataError(f"unknown placebo method {method!r}")

    observed_values = _mean_response(events, spec, series, horizon, benchmark_series)
    if observed_values is None:
        raise B4DataError(f"{spec.study_id}: no observable response at {horizon} to test")

    generator = random.Random(seed)
    placebo_effects: list[float] = []
    for _ in range(replicates):
        if method == "shift":
            resampled = shift_events_within_blocks(events, block_hours=block_hours, generator=generator)
        else:
            resampled = permute_labels_within_blocks(events, block_hours=block_hours, generator=generator)
        effect = _mean_response(resampled, spec, series, horizon, benchmark_series)
        if effect is not None:
            placebo_effects.append(effect)

    if not placebo_effects:
        raise B4DataError(f"{spec.study_id}: every placebo replicate produced no observations")

    stats = describe(placebo_effects)
    exceedance = sum(1 for effect in placebo_effects if abs(effect) >= abs(observed_values)) / len(
        placebo_effects
    )
    if exceedance <= 0.05:
        verdict = "the observed effect is larger than 95% of placebo effects"
    elif exceedance <= 0.20:
        verdict = "the observed effect is larger than most placebo effects, but not decisively"
    else:
        verdict = (
            f"placebo events produce an effect this large {exceedance:.0%} of the time: "
            "the procedure manufactures effects of this size from nothing"
        )

    return PlaceboResult(
        horizon=horizon,
        method=method,
        replicates=len(placebo_effects),
        seed=seed,
        observed_effect=observed_values,
        placebo_mean=stats.mean or 0.0,
        placebo_std=stats.std or 0.0,
        exceedance_rate=exceedance,
        placebo_quantiles=stats.quantiles,
        verdict=verdict,
    )


def _mean_response(
    events: Sequence[StudyEvent],
    spec: EventStudySpec,
    series: MarketSeries,
    horizon: str,
    benchmark_series: MarketSeries | None,
) -> float | None:
    """Mean response at one horizon, through the study's own machinery.

    Reusing filter/cluster/build rather than a shortcut is the point: a placebo
    that skipped clustering would be testing a different estimator than the one
    whose result it is meant to validate.
    """
    eligible = filter_events(events, spec)
    clusters = cluster_events(eligible, spec)
    observations = build_observations(clusters, spec, series, benchmark_series)
    values = [value for item in observations if (value := item.responses.get(horizon)) is not None]
    return sum(values) / len(values) if values else None


# ------------------------------------------------------------------ matching


def realized_volatility_before(series: MarketSeries, origin: datetime, lookback_hours: int = 24) -> float | None:
    """Dispersion of returns in the window ending at the origin.

    Strictly backward-looking: the matching variable must be knowable at the
    origin, or the control group is chosen with hindsight.
    """
    start = origin - timedelta(hours=lookback_hours)
    start_index = last_available_index(series, start)
    end_index = last_available_index(series, origin)
    if start_index < 0 or end_index - start_index < 2:
        return None
    closes = [bar.close for bar in series.bars[start_index : end_index + 1]]
    returns = [later / earlier - 1.0 for earlier, later in zip(closes, closes[1:], strict=False)]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return float(variance**0.5)


def matched_control_test(
    event_origins: Sequence[datetime],
    candidate_origins: Sequence[datetime],
    series: MarketSeries,
    horizon: str,
    *,
    volatility_buckets: int = 4,
    exclusion_hours: int = 72,
    seed: int = 20260831,
    benchmark_series: MarketSeries | None = None,
) -> MatchedControlResult:
    """Pair each event origin with a non-event origin from the same era and regime.

    Candidates within `exclusion_hours` of any event are removed first: a
    "non-event" origin one hour after an event is contaminated by it, and
    matching against contaminated controls is how a real effect gets explained
    away.
    """
    if not event_origins:
        raise B4DataError("matched controls need at least one event origin")

    events_sorted = sorted(event_origins)
    exclusion = timedelta(hours=exclusion_hours)
    clean_candidates = [
        candidate
        for candidate in sorted(candidate_origins)
        if all(abs(candidate - event) > exclusion for event in events_sorted)
    ]
    if not clean_candidates:
        raise B4DataError("every candidate control origin is within the exclusion window of an event")

    volatilities = {
        origin: realized_volatility_before(series, origin)
        for origin in list(events_sorted) + clean_candidates
    }
    observed = [value for value in volatilities.values() if value is not None]
    if not observed:
        raise B4DataError("no origin has enough history to compute a matching volatility")
    ordered = sorted(observed)
    edges = [
        ordered[min(len(ordered) - 1, int(len(ordered) * (index + 1) / volatility_buckets))]
        for index in range(volatility_buckets - 1)
    ]

    def bucket(origin: datetime) -> tuple[int, int, int] | None:
        volatility = volatilities.get(origin)
        if volatility is None:
            return None
        index = sum(1 for edge in edges if volatility > edge)
        return (origin.year, origin.month // 4, index)

    pool: dict[tuple[int, int, int], list[datetime]] = {}
    for candidate in clean_candidates:
        key = bucket(candidate)
        if key is not None:
            pool.setdefault(key, []).append(candidate)

    generator = random.Random(seed)
    for members in pool.values():
        generator.shuffle(members)

    matches: list[_Match] = []
    unmatched = 0
    for event_origin in events_sorted:
        key = bucket(event_origin)
        available = pool.get(key) if key is not None else None
        if not available:
            unmatched += 1
            continue
        matches.append(_Match(event_origin=event_origin, control_origin=available.pop()))

    event_values: list[float] = []
    control_values: list[float] = []
    for match in matches:
        event_value = event_response(series, match.event_origin, [horizon], benchmark_series)[horizon]
        control_value = event_response(series, match.control_origin, [horizon], benchmark_series)[horizon]
        if event_value is None or control_value is None:
            continue
        event_values.append(event_value)
        control_values.append(control_value)

    event_stats = describe(event_values)
    control_stats = describe(control_values)
    difference = (
        event_stats.mean - control_stats.mean
        if event_stats.mean is not None and control_stats.mean is not None
        else None
    )

    standard_error: float | None = None
    if event_values and event_stats.std is not None and control_stats.std is not None:
        pairs = len(event_values)
        standard_error = ((event_stats.std**2 + control_stats.std**2) / pairs) ** 0.5

    if not event_values:
        verdict = "INSUFFICIENT — no matched pair had an observable outcome at this horizon"
    elif difference is None or standard_error is None:
        verdict = "INSUFFICIENT — one side of the comparison is empty"
    elif standard_error == 0.0:
        verdict = (
            "event origins behave like their matched non-event controls"
            if difference == 0.0
            else "event origins differ from matched non-event controls"
        )
    elif abs(difference) < 2.0 * standard_error:
        verdict = "event origins behave like their matched non-event controls"
    else:
        verdict = (
            f"event origins differ from matched non-event controls by "
            f"{abs(difference) / standard_error:.1f} standard errors"
        )

    return MatchedControlResult(
        horizon=horizon,
        matched_pairs=len(event_values),
        unmatched_events=unmatched,
        event_stats=event_stats,
        control_stats=control_stats,
        mean_difference=difference,
        difference_standard_error=standard_error,
        verdict=verdict,
    )


def non_event_origins(
    series: MarketSeries,
    event_origins: Sequence[datetime],
    *,
    step_hours: int = 24,
    exclusion_hours: int = 72,
) -> list[datetime]:
    """Regularly spaced origins that are not near any event."""
    if series.start is None or series.end is None:
        raise B4DataError("cannot draw control origins from an empty series")
    exclusion = timedelta(hours=exclusion_hours)
    step = timedelta(hours=step_hours)
    origins: list[datetime] = []
    cursor = series.start + step
    while cursor <= series.end:
        if close_as_of(series, cursor) is not None and all(
            abs(cursor - event) > exclusion for event in event_origins
        ):
            origins.append(cursor)
        cursor += step
    return origins


__all__ = [
    "DEFAULT_PLACEBO_REPLICATES",
    "MatchedControlResult",
    "PlaceboResult",
    "matched_control_test",
    "non_event_origins",
    "permute_labels_within_blocks",
    "placebo_test",
    "realized_volatility_before",
    "shift_events_within_blocks",
]
