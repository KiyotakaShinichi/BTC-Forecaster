"""B4.25 – B4.31 — decay, stability, and what a result is actually made of.

An aggregate effect can be real, or it can be one extraordinary week wearing a
sample size. These checks separate the two, and every one of them is capable of
demoting a result that the headline statistics liked.

**Decay (B4.25)** is reported across horizons and never assumed monotone. Real
signals sometimes peak at six hours; forcing a decreasing curve onto the numbers
would hide that.

**Period stability (B4.26)** and **leave-one-period-out (B4.29)** ask whether one
era created the whole thing. Periods are broad on purpose — thin slices produce
unstable estimates that look like instability in the signal rather than in the
slicing.

**Event concentration (B4.28)** is the bluntest and often the most informative:
drop the single largest contributor and see what survives. A result that
evaporates was a description of one event.

**Entity and source concentration (B4.30, B4.31)** ask who the result is really
about. "ENTITY_STATEMENT looks useful" means something quite different when 90%
of the events are one person, or one publisher.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from .contracts import B4DataError
from .stats import describe


class DecayPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: str
    n: int
    mean: float | None
    #: Mean at this horizon divided by the mean at the first horizon. None when
    #: the reference is zero -- a ratio to zero is not a decay rate.
    ratio_to_first: float | None


class DecayProfile(BaseModel):
    """B4.25. The shape of a signal across horizons, described not fitted."""

    model_config = ConfigDict(frozen=True)

    signal_id: str
    points: tuple[DecayPoint, ...]
    monotone_decreasing: bool
    peak_horizon: str | None
    note: str


class PeriodResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    n: int
    mean: float | None
    std: float | None


class StabilityResult(BaseModel):
    """B4.26 / B4.29. Whether one era is doing all the work."""

    model_config = ConfigDict(frozen=True)

    periods: tuple[PeriodResult, ...]
    leave_one_period_out: tuple[PeriodResult, ...]
    full_sample_mean: float | None
    sign_consistent: bool
    #: Largest absolute change in the full-sample mean caused by dropping any
    #: single period, as a fraction of the full-sample mean.
    max_relative_shift: float | None
    stable: bool
    verdict: str


class ConcentrationResult(BaseModel):
    """B4.28 / B4.30 / B4.31. What the result is made of."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    n: int
    full_sample_mean: float | None
    top_contributors: tuple[tuple[str, float], ...]
    bottom_contributors: tuple[tuple[str, float], ...]
    leave_one_out_min: float | None
    leave_one_out_max: float | None
    #: True when removing one observation flips the sign or halves the estimate.
    fragile: bool
    largest_share: float | None
    largest_share_key: str | None
    verdict: str


# ---------------------------------------------------------------------- decay


def decay_profile(
    signal_id: str,
    values_by_horizon: Mapping[str, Sequence[float]],
    horizon_order: Sequence[str],
) -> DecayProfile:
    """B4.25. Describe how an effect changes across declared horizons."""
    points: list[DecayPoint] = []
    reference: float | None = None
    for horizon in horizon_order:
        stats = describe(values_by_horizon.get(horizon, []))
        if reference is None and stats.mean is not None:
            reference = stats.mean
        ratio = (
            stats.mean / reference
            if stats.mean is not None and reference is not None and reference != 0.0
            else None
        )
        points.append(DecayPoint(horizon=horizon, n=stats.n, mean=stats.mean, ratio_to_first=ratio))

    magnitudes = [(point.horizon, abs(point.mean)) for point in points if point.mean is not None]
    peak = max(magnitudes, key=lambda item: item[1])[0] if magnitudes else None
    monotone = all(
        earlier >= later for (_, earlier), (_, later) in zip(magnitudes, magnitudes[1:], strict=False)
    )

    if not magnitudes:
        note = "no horizon has an observable effect"
    elif monotone:
        note = "magnitude decreases across every declared horizon"
    else:
        note = (
            f"magnitude is not monotone; it peaks at {peak}. Decay is reported as measured "
            "rather than assumed"
        )

    return DecayProfile(
        signal_id=signal_id,
        points=tuple(points),
        monotone_decreasing=bool(magnitudes) and monotone,
        peak_horizon=peak,
        note=note,
    )


# ------------------------------------------------------------------ stability


def calendar_period(moment: datetime) -> str:
    """Broad period label. Calendar years: thin slices manufacture instability."""
    return str(moment.year)


def period_stability(
    observations: Sequence[tuple[datetime, float]],
    *,
    period_of: Callable[[datetime], str] = calendar_period,
    minimum_period_size: int = 20,
    shift_tolerance: float = 0.5,
) -> StabilityResult:
    """B4.26 and B4.29 together: per-period means and leave-one-period-out.

    Periods smaller than `minimum_period_size` are reported but excluded from
    the stability verdict. A three-observation year swinging wildly says nothing
    about the signal, and letting it drive the verdict would make every study
    look fragile.
    """
    if not observations:
        raise B4DataError("period stability needs at least one observation")

    grouped: dict[str, list[float]] = {}
    for moment, value in observations:
        grouped.setdefault(period_of(moment), []).append(value)

    periods = tuple(
        PeriodResult(
            label=label,
            n=describe(values).n,
            mean=describe(values).mean,
            std=describe(values).std,
        )
        for label, values in sorted(grouped.items())
    )

    all_values = [value for _, value in observations]
    full = describe(all_values)

    leave_out: list[PeriodResult] = []
    shifts: list[float] = []
    for label in sorted(grouped):
        remaining = [value for key, values in grouped.items() if key != label for value in values]
        stats = describe(remaining)
        leave_out.append(PeriodResult(label=f"without {label}", n=stats.n, mean=stats.mean, std=stats.std))
        if (
            stats.mean is not None
            and full.mean not in (None, 0.0)
            and len(grouped[label]) >= minimum_period_size
        ):
            assert full.mean is not None
            shifts.append(abs(stats.mean - full.mean) / abs(full.mean))

    considered = [period for period in periods if period.n >= minimum_period_size]
    means = [period.mean for period in considered if period.mean is not None]
    sign_consistent = bool(means) and (all(value > 0 for value in means) or all(value < 0 for value in means))
    max_shift = max(shifts) if shifts else None

    if len(considered) < 2:
        stable = False
        verdict = (
            f"INSUFFICIENT — fewer than two periods reach {minimum_period_size} observations, "
            "so stability cannot be assessed"
        )
    elif not sign_consistent:
        stable = False
        verdict = "UNSTABLE — the effect changes sign between periods"
    elif max_shift is not None and max_shift > shift_tolerance:
        stable = False
        verdict = (
            f"UNSTABLE — dropping one period moves the estimate by {max_shift:.0%} of its value"
        )
    else:
        stable = True
        verdict = "STABLE — the sign holds across periods and no single period dominates"

    return StabilityResult(
        periods=periods,
        leave_one_period_out=tuple(leave_out),
        full_sample_mean=full.mean,
        sign_consistent=sign_consistent,
        max_relative_shift=max_shift,
        stable=stable,
        verdict=verdict,
    )


# -------------------------------------------------------------- concentration


def concentration(
    labelled_values: Sequence[tuple[str, float]],
    *,
    dimension: str,
    top_k: int = 5,
    fragility_ratio: float = 0.5,
) -> ConcentrationResult:
    """B4.28 / B4.30 / B4.31: contributors, leave-one-out, and share.

    `fragile` is set when removing a *single* observation flips the sign of the
    estimate or moves it by more than `fragility_ratio` of its value. That is
    deliberately a low bar to fail: a result that cannot survive one deletion
    should never be described as a signal.
    """
    if not labelled_values:
        raise B4DataError("concentration needs at least one observation")

    values = [value for _, value in labelled_values]
    full = describe(values)
    ordered = sorted(labelled_values, key=lambda item: item[1])

    leave_one_out: list[float] = []
    for index in range(len(values)):
        remaining = values[:index] + values[index + 1 :]
        if remaining:
            leave_one_out.append(sum(remaining) / len(remaining))

    minimum = min(leave_one_out) if leave_one_out else None
    maximum = max(leave_one_out) if leave_one_out else None

    fragile = False
    if full.mean is not None and full.mean != 0.0 and minimum is not None and maximum is not None:
        sign_flip = (minimum < 0.0 < full.mean) or (maximum > 0.0 > full.mean)
        biggest_move = max(abs(minimum - full.mean), abs(maximum - full.mean)) / abs(full.mean)
        fragile = sign_flip or biggest_move > fragility_ratio

    counts = Counter(label for label, _ in labelled_values)
    largest_key, largest_count = counts.most_common(1)[0]
    largest_share = largest_count / len(labelled_values)

    if fragile:
        verdict = f"FRAGILE — removing one observation moves or flips the {dimension} estimate"
    elif largest_share >= 0.5:
        verdict = (
            f"CONCENTRATED — {largest_share:.0%} of observations come from {largest_key!r}; "
            f"the result is about {largest_key!r} more than about the {dimension} as a whole"
        )
    else:
        verdict = f"DIVERSE — no single {dimension} contributes half the observations, and no single observation drives the estimate"

    return ConcentrationResult(
        dimension=dimension,
        n=len(labelled_values),
        full_sample_mean=full.mean,
        top_contributors=tuple((label, value) for label, value in ordered[-top_k:][::-1]),
        bottom_contributors=tuple((label, value) for label, value in ordered[:top_k]),
        leave_one_out_min=minimum,
        leave_one_out_max=maximum,
        fragile=fragile,
        largest_share=largest_share,
        largest_share_key=largest_key,
        verdict=verdict,
    )


# ----------------------------------------------------------------- regimes


class RegimeStratification(BaseModel):
    """B4.27. Simple, causal, and refused outright when cells are too thin."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    cells: tuple[PeriodResult, ...]
    adequate: bool
    verdict: str


def stratify_by_regime(
    labelled: Sequence[tuple[str, float]],
    *,
    dimension: str,
    minimum_cell_size: int = 30,
) -> RegimeStratification:
    """Split an estimate by a regime label computable at the origin.

    No hidden-state model is fitted here (B4.27 forbids it, and rightly: a
    regime model estimated on the full sample would use the future to label the
    past). The label must be something a caller could have computed at the
    origin -- a trailing volatility bucket, a trailing trend sign.
    """
    grouped: dict[str, list[float]] = {}
    for label, value in labelled:
        grouped.setdefault(label, []).append(value)

    cells = tuple(
        PeriodResult(label=label, n=describe(values).n, mean=describe(values).mean, std=describe(values).std)
        for label, values in sorted(grouped.items())
    )
    adequate = len(cells) >= 2 and all(cell.n >= minimum_cell_size for cell in cells)
    verdict = (
        f"stratified across {len(cells)} regime cells"
        if adequate
        else f"INSUFFICIENT FOR REGIME ANALYSIS — needs at least two cells of {minimum_cell_size}+ observations"
    )
    return RegimeStratification(dimension=dimension, cells=cells, adequate=adequate, verdict=verdict)


def summarise_stability(
    stability: StabilityResult | None, event_concentration: ConcentrationResult | None
) -> bool | None:
    """Collapse the stability checks into the tri-state `classify` expects.

    None means "not assessed", which is different from "assessed and fine".
    Returning False for an unassessed signal would silently demote every study
    too small to check.
    """
    if stability is None and event_concentration is None:
        return None
    if stability is not None and not stability.stable:
        return False
    if event_concentration is not None and event_concentration.fragile:
        return False
    if stability is None or "INSUFFICIENT" in stability.verdict:
        return None
    return True


__all__ = [
    "ConcentrationResult",
    "DecayPoint",
    "DecayProfile",
    "PeriodResult",
    "RegimeStratification",
    "StabilityResult",
    "calendar_period",
    "concentration",
    "decay_profile",
    "period_stability",
    "stratify_by_regime",
    "summarise_stability",
]
