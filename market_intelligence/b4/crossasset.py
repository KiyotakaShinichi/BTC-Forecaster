"""B4.21 – B4.24 — cross-asset features, lead/lag, correlation, Granger gate.

The one domain where B4 has real point-in-time-valid history, so it is the one
place a genuine result could come from. That makes the discipline matter more,
not less.

**Lags are declared, not scanned (B4.22).** Sweeping 500 lags and reporting the
best one is a search over 500 hypotheses dressed up as a single finding. The lag
set is small, fixed in the preregistration, and every lag stays in the results
table whatever it shows.

**The effect is in the outcome's units.** A correlation coefficient cannot be
compared against a practical threshold expressed in returns. The reported effect
is the slope of BTC's forward return on a *standardised* predictor: the expected
forward return change per one-standard-deviation move in the other asset. The
correlation is reported alongside for context, never as the headline.

**Nothing here is causal (B4.23).** These are contemporaneous associations
between a past window and a future window. The language stays observational
throughout, and the Granger gate below refuses to run when its assumptions are
not met rather than lending the output unearned authority.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .contracts import B4DataError, MarketSeries
from .market_data import last_available_index
from .stats import BootstrapConfig, DescriptiveStats, TestRecord, block_bootstrap, describe
from .targets import horizon_delta


class CrossAssetFeatureKind(str, Enum):
    """B4.21. Every one is computed strictly from data available at the origin."""

    LAGGED_RETURN = "LAGGED_RETURN"
    ROLLING_RETURN = "ROLLING_RETURN"
    REALIZED_VOLATILITY = "REALIZED_VOLATILITY"
    RELATIVE_STRENGTH = "RELATIVE_STRENGTH"
    ZSCORED_MOVE = "ZSCORED_MOVE"


class CrossAssetFeatureSpec(BaseModel):
    """One declared feature of one asset."""

    model_config = ConfigDict(frozen=True)

    feature_id: str
    series_id: str
    kind: CrossAssetFeatureKind
    #: Window ending at (and including only data available at) the origin.
    window: str = "1d"
    #: Lookback for the z-score / relative-strength normalisation.
    normalisation_window: str = "30d"

    def label(self) -> str:
        return f"{self.series_id}:{self.kind.value}:{self.window}"


class LeadLagSpec(BaseModel):
    """B4.22. The whole declared lead/lag study."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    features: tuple[CrossAssetFeatureSpec, ...]
    horizons: tuple[str, ...] = ("1d", "3d", "7d")
    minimum_observations: int = Field(default=100, ge=2)
    bootstrap: BootstrapConfig = BootstrapConfig()
    family: str = "cross_asset"


class LeadLagResult(BaseModel):
    """One (feature, horizon) cell. Every declared cell is reported."""

    model_config = ConfigDict(frozen=True)

    feature_id: str
    series_id: str
    kind: CrossAssetFeatureKind
    window: str
    horizon: str
    n: int
    #: Forward-return change per one-sd move in the predictor, in return units.
    slope_per_sd: float | None
    correlation: float | None
    lower: float | None
    upper: float | None
    p_value: float | None
    predictor_stats: DescriptiveStats
    outcome_stats: DescriptiveStats
    insufficient: bool
    note: str = ""

    def as_test_record(self, family: str) -> TestRecord | None:
        if self.insufficient or self.slope_per_sd is None:
            return None
        assert self.lower is not None and self.upper is not None and self.p_value is not None
        return TestRecord(
            test_id=f"{self.feature_id}@{self.horizon}",
            family=family,
            n=self.n,
            effect=self.slope_per_sd,
            lower=self.lower,
            upper=self.upper,
            p_value=self.p_value,
        )


# ------------------------------------------------------------------ features


def _window_delta(window: str) -> timedelta:
    return horizon_delta(window)


def _return_over(series: MarketSeries, origin: datetime, window: str) -> float | None:
    """Return over the window ending at the origin, from available closes only."""
    start_index = last_available_index(series, origin - _window_delta(window))
    end_index = last_available_index(series, origin)
    if start_index < 0 or end_index <= start_index:
        return None
    return series.bars[end_index].close / series.bars[start_index].close - 1.0


def _returns_in_window(series: MarketSeries, origin: datetime, window: str) -> list[float]:
    start_index = last_available_index(series, origin - _window_delta(window))
    end_index = last_available_index(series, origin)
    if start_index < 0 or end_index - start_index < 2:
        return []
    closes = [bar.close for bar in series.bars[start_index : end_index + 1]]
    return [later / earlier - 1.0 for earlier, later in zip(closes, closes[1:], strict=False)]


def compute_feature(
    spec: CrossAssetFeatureSpec,
    series: MarketSeries,
    origin: datetime,
    reference: MarketSeries | None = None,
) -> float | None:
    """One feature value at one origin, or None when it is not computable.

    Every branch reads through `last_available_index`, so no feature can see a
    close that had not printed at the origin. That is the only invariant this
    function has, and it is the reason it does not take raw arrays.
    """
    if spec.kind is CrossAssetFeatureKind.LAGGED_RETURN:
        return _return_over(series, origin, spec.window)

    if spec.kind is CrossAssetFeatureKind.ROLLING_RETURN:
        return _return_over(series, origin, spec.window)

    if spec.kind is CrossAssetFeatureKind.REALIZED_VOLATILITY:
        returns = _returns_in_window(series, origin, spec.window)
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        return math.sqrt(variance)

    if spec.kind is CrossAssetFeatureKind.RELATIVE_STRENGTH:
        if reference is None:
            raise B4DataError(f"{spec.feature_id}: RELATIVE_STRENGTH needs a reference series")
        own = _return_over(series, origin, spec.window)
        other = _return_over(reference, origin, spec.window)
        return None if own is None or other is None else own - other

    if spec.kind is CrossAssetFeatureKind.ZSCORED_MOVE:
        recent = _return_over(series, origin, spec.window)
        history = _returns_in_window(series, origin, spec.normalisation_window)
        if recent is None or len(history) < 10:
            return None
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / len(history)
        sigma = math.sqrt(variance)
        return None if sigma == 0.0 else (recent - mean) / sigma

    raise B4DataError(f"unhandled feature kind {spec.kind}")  # pragma: no cover - enum is exhaustive


# ------------------------------------------------------------------- lead/lag


def _standardise(values: Sequence[float]) -> tuple[list[float], float] | None:
    count = len(values)
    if count < 2:
        return None
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    sigma = math.sqrt(variance)
    if sigma == 0.0:
        return None
    return [(value - mean) / sigma for value in values], sigma


def _slope_per_sd(predictors: Sequence[float], outcomes: Sequence[float]) -> float | None:
    """OLS slope of outcome on a standardised predictor.

    Standardising first is what makes the number readable: it is the expected
    forward-return change per one-sd move, so it can be compared directly
    against a practical threshold stated in return units.
    """
    standardised = _standardise(predictors)
    if standardised is None:
        return None
    scaled, _ = standardised
    mean_outcome = sum(outcomes) / len(outcomes)
    numerator = sum(x * (y - mean_outcome) for x, y in zip(scaled, outcomes, strict=True))
    denominator = sum(x * x for x in scaled)
    return None if denominator == 0.0 else numerator / denominator


def _correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 2:
        return None
    mean_a = sum(first) / len(first)
    mean_b = sum(second) / len(second)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second, strict=True))
    variance_a = sum((a - mean_a) ** 2 for a in first)
    variance_b = sum((b - mean_b) ** 2 for b in second)
    if variance_a == 0.0 or variance_b == 0.0:
        return None
    return covariance / math.sqrt(variance_a * variance_b)


def lead_lag_study(
    spec: LeadLagSpec,
    btc: MarketSeries,
    series_by_id: dict[str, MarketSeries],
    origins: Sequence[datetime],
    *,
    reference_id: str | None = None,
) -> list[LeadLagResult]:
    """Every declared (feature, horizon) cell, reported whatever it shows."""
    reference = series_by_id.get(reference_id) if reference_id else None
    results: list[LeadLagResult] = []

    for feature in spec.features:
        series = series_by_id.get(feature.series_id)
        if series is None:
            raise B4DataError(f"{feature.feature_id}: no series {feature.series_id!r}")

        for horizon in spec.horizons:
            predictors: list[float] = []
            outcomes: list[float] = []
            for origin in origins:
                predictor = compute_feature(feature, series, origin, reference)
                if predictor is None:
                    continue
                outcome = forward_return(btc, origin, horizon)
                if outcome is None:
                    continue
                predictors.append(predictor)
                outcomes.append(outcome)

            count = len(predictors)
            if count < spec.minimum_observations:
                results.append(
                    LeadLagResult(
                        feature_id=feature.feature_id,
                        series_id=feature.series_id,
                        kind=feature.kind,
                        window=feature.window,
                        horizon=horizon,
                        n=count,
                        slope_per_sd=None,
                        correlation=None,
                        lower=None,
                        upper=None,
                        p_value=None,
                        predictor_stats=describe(predictors),
                        outcome_stats=describe(outcomes),
                        insufficient=True,
                        note=f"{count} paired observations is below the declared minimum of "
                        f"{spec.minimum_observations}",
                    )
                )
                continue

            slope = _slope_per_sd(predictors, outcomes)
            if slope is None:
                results.append(
                    LeadLagResult(
                        feature_id=feature.feature_id,
                        series_id=feature.series_id,
                        kind=feature.kind,
                        window=feature.window,
                        horizon=horizon,
                        n=count,
                        slope_per_sd=None,
                        correlation=None,
                        lower=None,
                        upper=None,
                        p_value=None,
                        predictor_stats=describe(predictors),
                        outcome_stats=describe(outcomes),
                        insufficient=True,
                        note="the predictor has no variation at these origins",
                    )
                )
                continue

            # Bootstrap the slope by resampling *paired* observations in blocks,
            # so the serial dependence of overlapping forward windows survives.
            #
            # The predictor is standardised once, on the full sample, and not
            # re-standardised inside each replicate. Two reasons: re-standardising
            # would give every replicate its own units, so the resampled slopes
            # would not all be estimating the same quantity; and with the scaling
            # fixed the statistic collapses to a ratio of two sums, which is what
            # makes 63 cells x 1,000 replicates x ~3,000 observations tractable
            # in Python at all.
            standardised = _standardise(predictors)
            assert standardised is not None  # `slope` above would have been None
            scaled, _ = standardised
            cross = [x * y for x, y in zip(scaled, outcomes, strict=True)]
            square = [x * x for x in scaled]

            def slope_of(
                indices: Sequence[float],
                numerator: list[float] = cross,
                denominator: list[float] = square,
            ) -> float:
                top = 0.0
                bottom = 0.0
                for index in indices:
                    position = int(index)
                    top += numerator[position]
                    bottom += denominator[position]
                return top / bottom if bottom else 0.0

            bootstrap = block_bootstrap(
                [float(index) for index in range(count)], spec.bootstrap, statistic=slope_of
            )

            results.append(
                LeadLagResult(
                    feature_id=feature.feature_id,
                    series_id=feature.series_id,
                    kind=feature.kind,
                    window=feature.window,
                    horizon=horizon,
                    n=count,
                    slope_per_sd=slope,
                    correlation=_correlation(predictors, outcomes),
                    lower=bootstrap.lower,
                    upper=bootstrap.upper,
                    p_value=bootstrap.p_value,
                    predictor_stats=describe(predictors),
                    outcome_stats=describe(outcomes),
                    insufficient=False,
                )
            )

    return results


def forward_return(series: MarketSeries, origin: datetime, horizon: str) -> float | None:
    """BTC return from the close available at the origin to the one at origin+H."""
    start_index = last_available_index(series, origin)
    end = origin + horizon_delta(horizon)
    end_index = last_available_index(series, end)
    if start_index < 0 or end_index <= start_index:
        return None
    if series.end is not None and end > series.end:
        return None
    return series.bars[end_index].close / series.bars[start_index].close - 1.0


# ------------------------------------------------------------ Granger gate


class GrangerDecision(BaseModel):
    """B4.24. Whether the assumptions justify running it at all."""

    model_config = ConfigDict(frozen=True)

    run: bool
    reason: str
    observations: int
    max_lag: int
    stationarity_checked: bool
    predictor_stationary: bool | None = None
    outcome_stationary: bool | None = None


def granger_gate(
    predictor: Sequence[float],
    outcome: Sequence[float],
    *,
    max_lag: int = 5,
    minimum_observations: int = 200,
) -> GrangerDecision:
    """Decide whether a Granger-style test is defensible on this sample.

    Deliberately a gate rather than a test. "Granger causality" carries more
    authority than the procedure earns, and running it on a short or
    non-stationary sample launders a weak association into a causal-sounding
    claim. If the assumptions do not hold, the honest output is that it was not
    run and why.
    """
    count = min(len(predictor), len(outcome))
    if count < minimum_observations:
        return GrangerDecision(
            run=False,
            reason=(
                f"{count} observations is below the {minimum_observations} needed for a "
                f"{max_lag}-lag VAR to be estimated with any precision"
            ),
            observations=count,
            max_lag=max_lag,
            stationarity_checked=False,
        )
    if count < 10 * max_lag:
        return GrangerDecision(
            run=False,
            reason=f"{count} observations is fewer than ten per estimated lag",
            observations=count,
            max_lag=max_lag,
            stationarity_checked=False,
        )

    predictor_stationary = _looks_stationary(predictor)
    outcome_stationary = _looks_stationary(outcome)
    if not (predictor_stationary and outcome_stationary):
        return GrangerDecision(
            run=False,
            reason=(
                "at least one series fails the stationarity screen; a Granger test on "
                "non-stationary inputs reports spurious rejections"
            ),
            observations=count,
            max_lag=max_lag,
            stationarity_checked=True,
            predictor_stationary=predictor_stationary,
            outcome_stationary=outcome_stationary,
        )

    return GrangerDecision(
        run=True,
        reason="sample size and stationarity screen are both satisfied",
        observations=count,
        max_lag=max_lag,
        stationarity_checked=True,
        predictor_stationary=True,
        outcome_stationary=True,
    )


class GrangerResult(BaseModel):
    """B4.24. One pre-specified Granger-style test, in observational language."""

    model_config = ConfigDict(frozen=True)

    predictor_id: str
    outcome_id: str
    max_lag: int
    observations: int
    #: Smallest p-value across the tested lags, and the lag it came from. The
    #: minimum over lags is itself a search, so the lag count is carried into
    #: the multiple-testing family rather than quietly forgotten.
    best_lag: int
    p_value: float
    lag_p_values: dict[int, float]
    note: str

    def as_test_record(self, family: str) -> TestRecord:
        """Effect is left at zero: a Granger F-test has no effect size in the
        outcome's units, so it is carried for its p-value alone and can never
        clear a practical-significance threshold on its own."""
        return TestRecord(
            test_id=f"granger:{self.predictor_id}->{self.outcome_id}",
            family=family,
            n=self.observations,
            effect=0.0,
            lower=0.0,
            upper=0.0,
            p_value=self.p_value,
        )


def granger_study(
    predictor_id: str,
    outcome_id: str,
    predictor: Sequence[float],
    outcome: Sequence[float],
    *,
    max_lag: int = 5,
    minimum_observations: int = 200,
) -> tuple[GrangerDecision, GrangerResult | None]:
    """Run a bounded Granger-style test, but only if the gate allows it.

    Returns the gate decision alongside the result so a refusal is recorded as
    a finding rather than as a missing row. statsmodels is imported lazily; no
    unit test needs it.
    """
    decision = granger_gate(
        predictor, outcome, max_lag=max_lag, minimum_observations=minimum_observations
    )
    if not decision.run:
        return decision, None

    import numpy  # noqa: PLC0415
    from statsmodels.tsa.stattools import grangercausalitytests  # noqa: PLC0415

    count = min(len(predictor), len(outcome))
    # statsmodels tests whether column 1 Granger-causes column 0.
    data = numpy.column_stack([numpy.asarray(outcome[:count]), numpy.asarray(predictor[:count])])
    raw = grangercausalitytests(data, maxlag=max_lag)
    lag_p_values = {lag: float(raw[lag][0]["ssr_ftest"][1]) for lag in range(1, max_lag + 1)}
    best_lag = min(lag_p_values, key=lambda lag: lag_p_values[lag])

    return decision, GrangerResult(
        predictor_id=predictor_id,
        outcome_id=outcome_id,
        max_lag=max_lag,
        observations=count,
        best_lag=best_lag,
        p_value=lag_p_values[best_lag],
        lag_p_values=lag_p_values,
        note=(
            "observational: this tests whether past values of the predictor improve a linear "
            "forecast of the outcome beyond the outcome's own past. It is not evidence of "
            "causation, and the minimum over lags is corrected as part of its family."
        ),
    )


def _looks_stationary(values: Sequence[float], threshold: float = 0.9) -> bool:
    """Cheap screen: lag-1 autocorrelation well below one.

    Not an ADF test. A returns series -- which is what B4 feeds in -- either
    passes this comfortably or has a problem worth looking at directly, and a
    borderline unit-root verdict is not something this track should be resting
    a decision on.
    """
    count = len(values)
    if count < 3:
        return False
    mean = sum(values) / count
    denominator = sum((value - mean) ** 2 for value in values)
    if denominator == 0.0:
        return False
    numerator = sum(
        (earlier - mean) * (later - mean) for earlier, later in zip(values, values[1:], strict=False)
    )
    return abs(numerator / denominator) < threshold


__all__ = [
    "CrossAssetFeatureKind",
    "CrossAssetFeatureSpec",
    "GrangerDecision",
    "GrangerResult",
    "LeadLagResult",
    "LeadLagSpec",
    "compute_feature",
    "forward_return",
    "granger_gate",
    "granger_study",
    "lead_lag_study",
]
