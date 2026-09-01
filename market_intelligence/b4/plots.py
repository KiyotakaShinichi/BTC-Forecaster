"""B4.44 — reproducible research plots, generated from structured results.

Every figure here is drawn from a results object, never from an intermediate
computed for the plot. If a chart and a table disagree, one of them was computed
twice, and the one that is easy to mis-derive is always the chart.

These are research figures: axes labelled in the outcome's units, confidence
intervals drawn wherever an estimate is drawn, and sample sizes on the chart
rather than in a caption. No price charts with entry markers, no equity curves,
no annotations that imply a trade — B4.47 rules those out, and the visual
grammar of a trading chart implies a claim this track has not earned.

matplotlib is imported lazily and forced onto the Agg backend, so importing B4
never opens a display and unit tests never import it at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .contracts import B4DataError
from .crossasset import LeadLagResult
from .eventstudy import EventStudyResult
from .stability import DecayProfile, StabilityResult


def _pyplot() -> Any:
    import matplotlib  # noqa: PLC0415 -- lazy: importing B4 must not need a display

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot  # noqa: PLC0415

    return pyplot


def _save(figure: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=140, bbox_inches="tight")
    _pyplot().close(figure)
    return target


def plot_event_response(result: EventStudyResult, path: str | Path) -> Path:
    """Mean response by horizon with its bootstrap interval.

    Horizons that could not be estimated are drawn as a gap with an "n=0" tick
    rather than dropped, so the axis shows what was asked as well as what was
    answered.
    """
    pyplot = _pyplot()
    horizons = list(result.horizons)
    cells = [result.horizons[horizon] for horizon in horizons]
    means = [cell.descriptive.mean for cell in cells]
    lowers = [cell.bootstrap.lower if cell.bootstrap is not None else None for cell in cells]
    uppers = [cell.bootstrap.upper if cell.bootstrap is not None else None for cell in cells]

    figure, axes = pyplot.subplots(figsize=(6.5, 4.0))
    positions = range(len(horizons))
    for position, mean, lower, upper in zip(positions, means, lowers, uppers, strict=True):
        if mean is None:
            continue
        if lower is not None and upper is not None:
            axes.vlines(position, lower, upper, color="#4a6fa5", linewidth=2.0)
        axes.plot([position], [mean], marker="o", color="#1b3a5c", markersize=6)

    axes.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(
        [f"{h}\nn={result.horizons[h].descriptive.n}" for h in horizons]
    )
    axes.set_ylabel("mean forward return")
    axes.set_title(
        f"{result.study_id} — {result.observation_count} observations "
        f"({result.effective_event_count} effective)"
    )
    return _save(figure, path)


def plot_effect_intervals(results: Sequence[LeadLagResult], path: str | Path, *, top: int = 25) -> Path:
    """Effect sizes with intervals, ordered by effect rather than by significance.

    Ordering by |effect| rather than by p-value is deliberate: a chart sorted by
    significance reads as a ranking of importance, which is exactly the reading
    a multiple-testing correction exists to prevent.
    """
    estimable = [result for result in results if not result.insufficient and result.slope_per_sd is not None]
    if not estimable:
        raise B4DataError("no estimable cells to plot")
    ordered = sorted(estimable, key=lambda r: abs(r.slope_per_sd or 0.0), reverse=True)[:top]
    ordered = list(reversed(ordered))

    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(7.5, max(3.5, 0.28 * len(ordered))))
    for position, result in enumerate(ordered):
        lower, upper = result.lower, result.upper
        if lower is not None and upper is not None:
            colour = "#1b3a5c" if (lower > 0 or upper < 0) else "#9aa5b1"
            axes.hlines(position, lower, upper, color=colour, linewidth=2.0)
        axes.plot([result.slope_per_sd], [position], marker="o", color="#93441c", markersize=5)

    axes.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    axes.set_yticks(range(len(ordered)))
    axes.set_yticklabels([f"{r.feature_id}@{r.horizon} (n={r.n})" for r in ordered], fontsize=7)
    axes.set_xlabel("forward return per 1 sd of predictor")
    axes.set_title("Cross-asset lead/lag effects, 95% block-bootstrap intervals")
    return _save(figure, path)


def plot_decay(profile: DecayProfile, path: str | Path) -> Path:
    """Effect magnitude across horizons. Never smoothed into a decay curve."""
    pyplot = _pyplot()
    points = [point for point in profile.points if point.mean is not None]
    if not points:
        raise B4DataError(f"{profile.signal_id}: no horizon has an observable effect to plot")

    figure, axes = pyplot.subplots(figsize=(6.0, 3.6))
    axes.plot(
        [point.horizon for point in points],
        [abs(point.mean or 0.0) for point in points],
        marker="o",
        color="#1b3a5c",
    )
    axes.set_ylabel("|mean effect|")
    axes.set_xlabel("horizon")
    axes.set_title(f"{profile.signal_id} — {profile.note}", fontsize=9)
    return _save(figure, path)


def plot_period_stability(result: StabilityResult, path: str | Path, *, title: str = "") -> Path:
    """Per-period means beside the full-sample mean."""
    pyplot = _pyplot()
    periods = [period for period in result.periods if period.mean is not None]
    if not periods:
        raise B4DataError("no period has an estimable mean to plot")

    figure, axes = pyplot.subplots(figsize=(6.5, 3.6))
    axes.bar(
        [period.label for period in periods],
        [period.mean or 0.0 for period in periods],
        color="#4a6fa5",
    )
    if result.full_sample_mean is not None:
        axes.axhline(
            result.full_sample_mean,
            color="#93441c",
            linewidth=1.2,
            linestyle="--",
            label=f"full sample {result.full_sample_mean:+.5f}",
        )
        axes.legend(fontsize=8)
    axes.axhline(0.0, color="#999999", linewidth=0.8)
    axes.set_ylabel("mean effect")
    axes.set_title(f"{title or 'Period stability'} — {result.verdict}", fontsize=9)
    return _save(figure, path)


def plot_coverage_over_time(
    origins: Sequence[Any], coverage: Sequence[float], path: str | Path
) -> Path:
    """B4.34. Intelligence coverage through time, so gaps are visible."""
    if len(origins) != len(coverage):
        raise B4DataError("coverage series and origins must be the same length")
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(7.5, 3.0))
    axes.fill_between(list(origins), list(coverage), color="#4a6fa5", alpha=0.5)
    axes.set_ylim(0.0, 1.05)
    axes.set_ylabel("provider coverage ratio")
    axes.set_title("Intelligence coverage over time")
    return _save(figure, path)


__all__ = [
    "plot_coverage_over_time",
    "plot_decay",
    "plot_effect_intervals",
    "plot_event_response",
    "plot_period_stability",
]
