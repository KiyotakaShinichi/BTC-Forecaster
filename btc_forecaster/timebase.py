"""The canonical time contract.

Every timestamp in this platform is timezone-aware UTC. Every observation knows
two distinct instants, and confusing them is how look-ahead bias enters a
backtest:

``event_time``
    The instant the observation *refers to*. For a daily close bar labelled
    ``2024-01-05``, the close is only determined at the end of that UTC day, so
    ``event_time`` is ``2024-01-06T00:00:00Z``.

``available_time``
    The instant the observation could first have been *acted on*.
    ``available_time = event_time + publication_lag``. For a spot price the lag
    is zero: the close is known the moment the bar ends. For anything published
    after the fact -- a revised index, an on-chain metric with settlement delay,
    or any future exogenous signal from Track B -- the lag is positive and must
    be declared.

A forecast made at ``forecast_origin`` may consume an observation if and only if
``available_time <= forecast_origin``. That single inequality is the whole
leakage rule, and :func:`assert_available` is the only place it is enforced.

Bar labelling convention
------------------------
A daily bar is labelled by the UTC midnight at which it *starts* and covers the
half-open interval ``[bar_start, bar_start + 1 day)``::

    label / bar_start   2024-01-05T00:00Z
    covers              [2024-01-05T00:00Z, 2024-01-06T00:00Z)
    event_time          2024-01-06T00:00Z      <- close determined here
    available_time      2024-01-06T00:00Z      <- lag 0 for spot price

This matches how ``yfinance`` labels daily bars, so no re-indexing is needed on
ingest -- only localisation to UTC.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

# UTC is imported here and re-exported via __all__ so the rest of the package has
# a single place to import it from; `from datetime import UTC` at every call site
# invites someone to reach for a naive default instead.

#: Bar labels are UTC midnights, one calendar day apart.
BAR_FREQ = "D"
BAR_DURATION = pd.Timedelta(days=1)

#: Spot prices are known the instant their bar closes.
ZERO_LAG = pd.Timedelta(0)


class PointInTimeViolation(AssertionError):
    """Raised when information would be used before it existed."""


def to_utc_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Coerce any scalar date-like to a tz-aware UTC ``Timestamp``.

    Naive input is *assumed* to already be UTC rather than local time. Daily
    crypto bars from every provider we support are UTC-based, and silently
    applying the machine's local timezone would shift bars by up to a day
    depending on where the run happened -- a reproducibility bug that is
    invisible until someone runs the pipeline in another region.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize(UTC) if ts.tz is None else ts.tz_convert(UTC)


def to_utc_index(index: Iterable) -> pd.DatetimeIndex:
    """Coerce an index to tz-aware UTC daily bar labels (normalised to midnight)."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(index) if not isinstance(index, pd.Index) else index))
    idx = idx.tz_localize(UTC) if idx.tz is None else idx.tz_convert(UTC)
    return idx.normalize()


def bar_event_time(bar_start: pd.Timestamp) -> pd.Timestamp:
    """The instant a bar's close is determined: the end of the bar."""
    return to_utc_timestamp(bar_start) + BAR_DURATION


def available_time(
    bar_start: pd.Timestamp,
    publication_lag: pd.Timedelta = ZERO_LAG,
) -> pd.Timestamp:
    """The instant an observation from ``bar_start`` may first be acted on."""
    if publication_lag < ZERO_LAG:
        raise ValueError("publication_lag must be non-negative; time does not run backwards")
    return bar_event_time(bar_start) + publication_lag


def assert_available(
    when_available: pd.Timestamp,
    origin: ForecastOrigin | pd.Timestamp,
    *,
    what: str = "observation",
) -> None:
    """Enforce the point-in-time rule, or raise :class:`PointInTimeViolation`."""
    origin_ts = origin.timestamp if isinstance(origin, ForecastOrigin) else to_utc_timestamp(origin)
    when = to_utc_timestamp(when_available)
    if when > origin_ts:
        raise PointInTimeViolation(
            f"{what} becomes available at {when.isoformat()}, "
            f"after the forecast origin {origin_ts.isoformat()}"
        )


@dataclass(frozen=True)
class ForecastOrigin:
    """The point in time a forecast is made from.

    An origin is anchored to the *last observed bar*. The forecast is issued at
    that bar's ``event_time`` -- the moment its close became known -- and may use
    nothing that was unavailable then.
    """

    last_observed_bar: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "last_observed_bar", to_utc_timestamp(self.last_observed_bar).normalize())

    @property
    def timestamp(self) -> pd.Timestamp:
        """The instant the forecast is issued."""
        return bar_event_time(self.last_observed_bar)

    @property
    def first_target_bar(self) -> pd.Timestamp:
        """The first bar being forecast: the one immediately after the origin."""
        return self.last_observed_bar + BAR_DURATION

    def target_bars(self, horizon: int) -> pd.DatetimeIndex:
        """Labels of the ``horizon`` bars this origin forecasts.

        ``horizon=1`` is the next bar. There is no zero-step forecast: step 0 is
        already observed.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        return pd.date_range(
            start=self.first_target_bar,
            periods=horizon,
            freq=BAR_FREQ,
            tz=UTC,
            name="date",
        )

    def observable(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Subset of ``index`` whose bars had closed by this origin."""
        idx = to_utc_index(index)
        return idx[idx <= self.last_observed_bar]

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"ForecastOrigin(last_bar={self.last_observed_bar.date()}, issued={self.timestamp.isoformat()})"


@dataclass(frozen=True)
class HorizonSpec:
    """What a model is being asked to predict.

    ``steps`` is the number of daily bars ahead. Keeping this a named object
    rather than a bare int means the walk-forward engine can hand every model
    the identical instruction, and metrics can be reported per-step.
    """

    steps: int

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"horizon steps must be >= 1, got {self.steps}")

    def __len__(self) -> int:
        return self.steps

    def bars_from(self, origin: ForecastOrigin) -> pd.DatetimeIndex:
        return origin.target_bars(self.steps)


def infer_origin(index: pd.DatetimeIndex) -> ForecastOrigin:
    """Take the last observed bar of a series as the forecast origin."""
    idx = to_utc_index(index)
    if len(idx) == 0:
        raise ValueError("cannot infer a forecast origin from an empty index")
    return ForecastOrigin(idx.max())


def is_regular_daily(index: pd.DatetimeIndex) -> bool:
    """Whether an index is a gapless run of consecutive UTC daily bars."""
    idx = to_utc_index(index)
    if len(idx) < 2:
        return True
    return bool((idx.to_series().diff().dropna() == BAR_DURATION).all())


def calendar_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((to_utc_timestamp(end).normalize() - to_utc_timestamp(start).normalize()) / BAR_DURATION)


__all__ = [
    "UTC",
    "BAR_FREQ",
    "BAR_DURATION",
    "ZERO_LAG",
    "PointInTimeViolation",
    "ForecastOrigin",
    "HorizonSpec",
    "assert_available",
    "available_time",
    "bar_event_time",
    "calendar_days_between",
    "infer_origin",
    "is_regular_daily",
    "to_utc_index",
    "to_utc_timestamp",
]
