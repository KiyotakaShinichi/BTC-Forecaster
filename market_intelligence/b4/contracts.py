"""B4.1 / B4.36 — the historical data contract and the two evidence tiers.

Everything in B4 hangs off one rule, inherited from B3.1:

    only information with ``available_at <= forecast_origin`` may inform an
    origin, and only information with a timestamp strictly after the origin may
    become a target.

For market series that rule needs a convention, because a bar's *label* and the
moment its contents became knowable are different things. A daily bar labelled
2026-08-30 covers a whole day; its close is not knowable until the day ends.
Treating the label as the availability time is the single easiest way to leak a
day of hindsight into an entire study, and it looks completely reasonable in
code.

So a bar carries both, and `available_at` is derived conservatively:

    available_at(close) = period_start + period_length

For a 24/7 UTC series (BTC) that is exact. For an exchange-local series the true
close lands *earlier* than the next period boundary — the NYSE close on a Friday
labelled 2026-08-28 is knowable at 20:00 UTC, while this rule says 04:00 UTC
Saturday. Later than the truth is the safe direction: it can only ever make a
study weaker, never falsely stronger. The alternative — modelling every
exchange's session calendar and holiday schedule — buys a few hours of freshness
in exchange for a large surface of silent, venue-specific leakage bugs. Not
worth it for a study that is asking whether an effect exists at all.

The evidence tier is the second half of the contract. A news article retrieved
today about an event in 2021 is not evidence that was available in 2021, however
accurate it is. B4 refuses to let that distinction be implicit.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..errors import IntelligenceError


class B4DataError(IntelligenceError):
    """A historical observation violates the B4 data contract.

    Deliberately *not* a `ValueError`. Pydantic converts `ValueError` raised
    inside a validator into a `ValidationError`, so the same logical violation
    would surface as two different exception types depending on whether it was
    detected inside a model or outside one -- and callers would have to catch
    both to be correct. Staying off `ValueError` makes it propagate unchanged.
    """


class LeakageError(IntelligenceError):
    """Information that was not available at an origin reached a feature.

    Deliberately not a subclass of ``ValueError``: leakage is never a
    recoverable input problem to be caught and defaulted, it is a result that
    must be thrown away.
    """


class EvidenceTier(str, Enum):
    """B4.36. Whether historical availability is defensible, or merely assumed."""

    #: Availability at the stated time can be defended from the source's own
    #: publication mechanics. Eligible for later out-of-sample forecasting work.
    PIT_VALIDATED = "PIT_VALIDATED"

    #: Known now; historical availability cannot be established. Usable for
    #: description only, never as forecasting evidence, and never pooled with
    #: PIT_VALIDATED data in the same study.
    RETROSPECTIVE_ONLY = "RETROSPECTIVE_ONLY"


class TimestampConvention(str, Enum):
    """B4.1. What a series' timestamp actually denotes."""

    PERIOD_OPEN = "PERIOD_OPEN"
    PERIOD_CLOSE = "PERIOD_CLOSE"
    OBSERVATION_PUBLICATION = "OBSERVATION_PUBLICATION"
    PROVIDER_AVAILABILITY = "PROVIDER_AVAILABILITY"


class DataAvailability(str, Enum):
    """B4.0. Whether a catalogued source can support point-in-time study."""

    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class SourceDomain(str, Enum):
    """B4.0 inventory categories."""

    BTC_MARKET = "BTC_MARKET"
    MACRO_MARKET = "MACRO_MARKET"
    CROSS_ASSET = "CROSS_ASSET"
    CRYPTO_MARKET_STRUCTURE = "CRYPTO_MARKET_STRUCTURE"
    NEWS_WEB = "NEWS_WEB"
    ENTITY_STATEMENT = "ENTITY_STATEMENT"
    REGULATORY_EVENT = "REGULATORY_EVENT"
    ETF_EVENT = "ETF_EVENT"
    ONCHAIN_WHALE = "ONCHAIN_WHALE"


def require_utc(value: datetime, field: str = "timestamp") -> datetime:
    """Reject naive datetimes rather than guessing.

    Same rule as the B3.1 origin generator, for the same reason: assuming UTC
    makes a dataset depend on the machine that built it, and the discrepancy
    shows up as a silent one-hour shift twice a year rather than as an error.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise B4DataError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


class MarketBar(BaseModel):
    """One OHLC period, carrying both its label and its availability."""

    model_config = ConfigDict(frozen=True)

    #: Start of the period the bar covers, in UTC.
    period_start: datetime
    #: Nominal length of the period. `available_at` is derived from it.
    period_seconds: int = Field(gt=0)
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0.0)

    @field_validator("period_start")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return require_utc(value, "period_start")

    @model_validator(mode="after")
    def _coherent(self) -> "MarketBar":
        if self.high < self.low:
            raise B4DataError("bar high is below its low")
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise B4DataError("bar open/close fall outside the high-low range")
        if self.close <= 0.0 or self.open <= 0.0:
            raise B4DataError("bar prices must be positive")
        return self

    @property
    def period(self) -> timedelta:
        return timedelta(seconds=self.period_seconds)

    @property
    def available_at(self) -> datetime:
        """When this bar's close became knowable. See the module docstring."""
        return self.period_start + self.period

    @property
    def period_end(self) -> datetime:
        return self.period_start + self.period


class MarketSeries(BaseModel):
    """An ordered, gap-tolerant price series with declared provenance.

    Gap-tolerant on purpose: an exchange-local series is *supposed* to have
    weekend and holiday gaps, and a contract that rejected them would force
    every caller to forward-fill before validation — which is exactly how a
    synthetic Sunday close ends up in a study.
    """

    model_config = ConfigDict(frozen=True)

    series_id: str
    ticker: str
    domain: SourceDomain
    provider: str
    timestamp_convention: TimestampConvention
    evidence_tier: EvidenceTier
    #: Timezone the provider's own labels were expressed in, before conversion.
    source_timezone: str
    bars: tuple[MarketBar, ...]
    #: Rows the provider returned that failed the bar contract and were
    #: quarantined rather than repaired. Provenance, not an error count to
    #: ignore: a series that silently dropped rows is a different dataset.
    rejected_bar_count: int = 0
    rejection_summary: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> "MarketSeries":
        starts = [bar.period_start for bar in self.bars]
        if any(later <= earlier for earlier, later in zip(starts, starts[1:], strict=False)):
            raise B4DataError(f"{self.series_id}: bars must be strictly increasing in period_start")
        periods = {bar.period_seconds for bar in self.bars}
        if len(periods) > 1:
            raise B4DataError(f"{self.series_id}: mixed bar periods {sorted(periods)}")
        return self

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def start(self) -> datetime | None:
        return self.bars[0].period_start if self.bars else None

    @property
    def end(self) -> datetime | None:
        return self.bars[-1].period_end if self.bars else None

    @property
    def period_seconds(self) -> int:
        return self.bars[0].period_seconds if self.bars else 0

    def fingerprint(self) -> str:
        """Content hash over the values a study actually consumes.

        Provenance metadata is included, because the same numbers fetched under
        a different timestamp convention are not the same dataset.
        """
        digest = hashlib.sha256()
        digest.update(
            "|".join(
                [
                    self.series_id,
                    self.ticker,
                    self.provider,
                    self.timestamp_convention.value,
                    self.evidence_tier.value,
                    self.source_timezone,
                    str(self.rejected_bar_count),
                ]
            ).encode()
        )
        for bar in self.bars:
            digest.update(
                f"{bar.period_start.isoformat()}|{bar.period_seconds}|"
                f"{bar.open!r}|{bar.high!r}|{bar.low!r}|{bar.close!r}|{bar.volume!r}".encode()
            )
        return digest.hexdigest()


__all__ = [
    "B4DataError",
    "DataAvailability",
    "EvidenceTier",
    "LeakageError",
    "MarketBar",
    "MarketSeries",
    "SourceDomain",
    "TimestampConvention",
    "require_utc",
]
