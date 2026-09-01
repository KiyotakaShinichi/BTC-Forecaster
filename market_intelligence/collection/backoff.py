"""B4.1.21 — classify a failure before deciding whether to retry it.

Retrying is only correct for some failures, and the expensive mistake is
uniform treatment. A 401 will fail identically forever, so retrying it in a
tight loop burns rate-limit budget and can get an operator's key suspended for
the sole reason that the code could not tell "wrong credential" from "server
briefly busy". A 429 needs a *longer* wait, not the standard one. A schema
change needs a human, not a sleep.

So: classify, then act. The classification is the interesting part; the sleep
schedule is ordinary exponential backoff with jitter, and the jitter is there
because several providers polled on one cron tick otherwise retry in lockstep.

`sleep` is injected so tests exercise real schedules without real waiting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

from ..errors import ProviderUnavailableError

T = TypeVar("T")


class FailureClass(str, Enum):
    """B4.1.21. What kind of failure this is, and therefore what to do."""

    #: Credentials rejected. Never retried: it will fail identically forever,
    #: and hammering an auth endpoint is how a key gets suspended.
    AUTH = "AUTH"
    #: Provider says slow down. Retried, with a longer wait than transient.
    RATE_LIMIT = "RATE_LIMIT"
    #: Network blip, timeout, 5xx. The one class ordinary backoff is for.
    TRANSIENT = "TRANSIENT"
    #: 404, malformed URL, provider retired the endpoint. Not retried.
    PERMANENT = "PERMANENT"
    #: Response parsed but did not match the contract. Not retried — the shape
    #: will not change on a second call, and a human needs to look.
    SCHEMA = "SCHEMA"
    #: Response was well-formed but unusable (empty body, missing article).
    #: Not retried; recorded so persistent emptiness is visible.
    CONTENT = "CONTENT"


RETRYABLE = (FailureClass.TRANSIENT, FailureClass.RATE_LIMIT)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded, declared, and deliberately unambitious."""

    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    #: Rate-limit responses wait this multiple of the transient delay.
    rate_limit_multiplier: float = 4.0
    jitter: bool = True
    seed: int | None = None

    def delay_for(self, attempt: int, failure: FailureClass, generator: random.Random) -> float:
        """Delay before attempt `attempt` (1-based), for this failure class."""
        exponential: float = self.base_delay_seconds * float(2 ** max(0, attempt - 1))
        if failure is FailureClass.RATE_LIMIT:
            exponential *= self.rate_limit_multiplier
        capped: float = min(exponential, self.max_delay_seconds)
        if not self.jitter:
            return capped
        # Full jitter. Equal-width jitter still leaves a synchronised floor,
        # which is exactly the thing that keeps cron-launched pollers in step.
        return float(generator.uniform(0.0, capped))


class ProviderFailure(ProviderUnavailableError):
    """A classified provider failure."""

    def __init__(self, failure_class: FailureClass, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


def classify_http_status(status: int) -> FailureClass:
    if status in (401, 403):
        return FailureClass.AUTH
    if status == 429:
        return FailureClass.RATE_LIMIT
    if status in (404, 410):
        return FailureClass.PERMANENT
    if 500 <= status < 600:
        return FailureClass.TRANSIENT
    if 400 <= status < 500:
        return FailureClass.PERMANENT
    return FailureClass.TRANSIENT


def classify_exception(error: BaseException) -> FailureClass:
    """Best-effort classification for an exception with no HTTP status.

    Falls back to TRANSIENT rather than PERMANENT: a wrongly-transient failure
    costs a couple of retries, a wrongly-permanent one silently drops a provider
    from the corpus for as long as nobody notices.
    """
    if isinstance(error, ProviderFailure):
        return error.failure_class
    status = getattr(error, "code", None) or getattr(error, "status", None)
    if isinstance(status, int):
        return classify_http_status(status)
    name = type(error).__name__.casefold()
    if "timeout" in name:
        return FailureClass.TRANSIENT
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return FailureClass.SCHEMA
    return FailureClass.TRANSIENT


@dataclass
class AttemptLog:
    """What actually happened, for provider health and the run manifest."""

    attempts: int = 0
    slept_seconds: float = 0.0
    failures: tuple[FailureClass, ...] = ()

    def record(self, failure: FailureClass) -> None:
        self.failures = (*self.failures, failure)


DEFAULT_RETRY = RetryPolicy()


def call_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] | None = None,
    log: AttemptLog | None = None,
) -> T:
    """Run `operation`, retrying only failures that retrying can fix.

    Raises the last `ProviderFailure` when attempts are exhausted, so the caller
    records a classified failure rather than a bare exception.
    """
    policy = policy or DEFAULT_RETRY
    generator = random.Random(policy.seed)
    record = log if log is not None else AttemptLog()
    waiter = sleep if sleep is not None else _real_sleep
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        record.attempts = attempt
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 -- classified immediately below
            failure = classify_exception(error)
            record.record(failure)
            last = error
            if failure not in RETRYABLE or attempt == policy.max_attempts:
                raise ProviderFailure(failure, f"{failure.value}: {error}") from error
            delay = policy.delay_for(attempt, failure, generator)
            record.slept_seconds += delay
            waiter(delay)

    raise ProviderFailure(FailureClass.TRANSIENT, f"exhausted retries: {last}")  # pragma: no cover


def _real_sleep(seconds: float) -> None:  # pragma: no cover - not exercised in tests
    import time

    time.sleep(seconds)


__all__ = [
    "DEFAULT_RETRY",
    "RETRYABLE",
    "AttemptLog",
    "FailureClass",
    "ProviderFailure",
    "RetryPolicy",
    "call_with_retry",
    "classify_exception",
    "classify_http_status",
]
