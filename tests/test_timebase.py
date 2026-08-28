"""The time contract is the foundation of every leakage guarantee, so it is
tested directly rather than only through the components that rely on it."""

from __future__ import annotations

import pandas as pd
import pytest

from btc_forecaster.timebase import (
    BAR_DURATION,
    UTC,
    ForecastOrigin,
    HorizonSpec,
    PointInTimeViolation,
    assert_available,
    available_time,
    bar_event_time,
    calendar_days_between,
    infer_origin,
    is_regular_daily,
    to_utc_index,
    to_utc_timestamp,
)


class TestUtcCoercion:
    def test_naive_input_is_treated_as_utc_not_local_time(self):
        """A naive timestamp must not shift depending on where the run happens."""
        ts = to_utc_timestamp("2024-01-05")
        assert ts.tz is not None
        assert str(ts.tz) == "UTC"
        assert ts.hour == 0
        assert ts.isoformat() == "2024-01-05T00:00:00+00:00"

    def test_aware_input_is_converted_not_relabelled(self):
        tokyo = pd.Timestamp("2024-01-05 09:00", tz="Asia/Tokyo")
        ts = to_utc_timestamp(tokyo)
        assert ts == pd.Timestamp("2024-01-05 00:00", tz=UTC)

    def test_index_is_normalised_to_midnight(self):
        idx = to_utc_index(pd.to_datetime(["2024-01-05 13:45", "2024-01-06 22:10"]))
        assert (idx == idx.normalize()).all()
        assert str(idx.tz) == "UTC"

    def test_coercion_is_idempotent(self):
        once = to_utc_index(pd.date_range("2024-01-01", periods=5, freq="D"))
        twice = to_utc_index(once)
        pd.testing.assert_index_equal(once, twice)


class TestBarSemantics:
    def test_event_time_is_the_end_of_the_bar_not_its_label(self):
        """A daily close is only determined once the day is over."""
        assert bar_event_time(pd.Timestamp("2024-01-05", tz=UTC)) == pd.Timestamp(
            "2024-01-06 00:00", tz=UTC
        )

    def test_spot_price_is_available_the_moment_its_bar_closes(self):
        bar = pd.Timestamp("2024-01-05", tz=UTC)
        assert available_time(bar) == bar_event_time(bar)

    def test_publication_lag_delays_availability(self):
        """The seam future exogenous signals will use."""
        bar = pd.Timestamp("2024-01-05", tz=UTC)
        lagged = available_time(bar, publication_lag=pd.Timedelta(hours=36))
        assert lagged == pd.Timestamp("2024-01-07 12:00", tz=UTC)
        assert lagged > bar_event_time(bar)

    def test_negative_publication_lag_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            available_time(pd.Timestamp("2024-01-05", tz=UTC), publication_lag=pd.Timedelta(days=-1))


class TestForecastOrigin:
    def test_origin_is_issued_when_its_last_bar_closes(self):
        origin = ForecastOrigin(pd.Timestamp("2024-01-05", tz=UTC))
        assert origin.timestamp == pd.Timestamp("2024-01-06 00:00", tz=UTC)

    def test_first_target_is_the_bar_after_the_origin(self):
        origin = ForecastOrigin("2024-01-05")
        assert origin.first_target_bar == pd.Timestamp("2024-01-06", tz=UTC)

    def test_target_bars_are_consecutive_utc_days(self):
        bars = ForecastOrigin("2024-01-05").target_bars(3)
        assert list(bars) == [
            pd.Timestamp("2024-01-06", tz=UTC),
            pd.Timestamp("2024-01-07", tz=UTC),
            pd.Timestamp("2024-01-08", tz=UTC),
        ]

    def test_target_bars_never_include_the_origin_bar(self):
        """Step 0 is already observed; forecasting it would be trivially leaky."""
        origin = ForecastOrigin("2024-01-05")
        assert origin.last_observed_bar not in set(origin.target_bars(10))

    @pytest.mark.parametrize("horizon", [0, -1])
    def test_non_positive_horizon_is_rejected(self, horizon):
        with pytest.raises(ValueError, match="horizon"):
            ForecastOrigin("2024-01-05").target_bars(horizon)

    def test_origin_normalises_and_is_hashable(self):
        a = ForecastOrigin(pd.Timestamp("2024-01-05 17:30", tz=UTC))
        b = ForecastOrigin("2024-01-05")
        assert a == b
        assert len({a, b}) == 1

    def test_observable_excludes_bars_after_the_origin(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC)
        origin = ForecastOrigin("2024-01-05")
        observable = origin.observable(index)
        assert observable.max() == pd.Timestamp("2024-01-05", tz=UTC)
        assert len(observable) == 5

    def test_infer_origin_takes_the_last_bar(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC)
        assert infer_origin(index) == ForecastOrigin("2024-01-10")

    def test_infer_origin_on_empty_index_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            infer_origin(pd.DatetimeIndex([], tz=UTC))


class TestPointInTimeRule:
    """`assert_available` is the single enforcement point for the leakage rule."""

    def test_information_from_before_the_origin_is_allowed(self):
        origin = ForecastOrigin("2024-01-05")
        assert_available(available_time(pd.Timestamp("2024-01-04", tz=UTC)), origin)

    def test_information_available_exactly_at_the_origin_is_allowed(self):
        """The origin's own close is usable: it is what defines the origin."""
        origin = ForecastOrigin("2024-01-05")
        assert_available(available_time(origin.last_observed_bar), origin)

    def test_information_from_after_the_origin_is_rejected(self):
        origin = ForecastOrigin("2024-01-05")
        with pytest.raises(PointInTimeViolation, match="after the forecast origin"):
            assert_available(available_time(pd.Timestamp("2024-01-06", tz=UTC)), origin, what="tomorrow's close")

    def test_a_lagged_signal_from_a_past_bar_can_still_be_unavailable(self):
        """The exact case a naive `bar <= origin` check gets wrong."""
        origin = ForecastOrigin("2024-01-05")
        signal_bar = pd.Timestamp("2024-01-05", tz=UTC)
        assert signal_bar <= origin.last_observed_bar  # a naive check would pass
        with pytest.raises(PointInTimeViolation):
            assert_available(available_time(signal_bar, pd.Timedelta(days=2)), origin)

    def test_violation_message_names_the_offender(self):
        origin = ForecastOrigin("2024-01-05")
        with pytest.raises(PointInTimeViolation, match="sentiment score"):
            assert_available(
                available_time(pd.Timestamp("2024-02-01", tz=UTC)), origin, what="sentiment score"
            )


class TestHorizonSpec:
    def test_horizon_length_and_bars_agree(self):
        spec = HorizonSpec(30)
        origin = ForecastOrigin("2024-01-05")
        assert len(spec) == 30
        assert len(spec.bars_from(origin)) == 30

    def test_zero_horizon_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="steps"):
            HorizonSpec(0)


class TestCalendarHelpers:
    def test_regular_daily_detection(self):
        assert is_regular_daily(pd.date_range("2024-01-01", periods=5, freq="D", tz=UTC))
        gappy = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-05"], tz=UTC)
        assert not is_regular_daily(gappy)

    def test_single_bar_counts_as_regular(self):
        assert is_regular_daily(pd.DatetimeIndex(["2024-01-01"], tz=UTC))

    def test_calendar_days_between_is_inclusive_of_neither_endpoint(self):
        assert calendar_days_between("2024-01-01", "2024-01-08") == 7

    def test_bar_duration_is_one_day(self):
        assert BAR_DURATION == pd.Timedelta(days=1)
