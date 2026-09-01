"""B4.1.6 — WhaleDataProvider, with UNKNOWN kept UNKNOWN.

The rule that shapes this module: **do not infer wallet ownership or exchange
attribution beyond what the provider actually evidenced.**

It is very tempting to do otherwise. A large transfer to an address that "looks
like" an exchange is trivially classified as an inflow, and the resulting
dataset has far fewer UNKNOWNs and far more apparent signal. It is also
fabricated. B4's whale hypotheses depend on inflow and outflow meaning what they
say, and a heuristic that guesses would be indistinguishable in the data from a
provider that knew.

So classification comes from the provider's own attribution or it is UNKNOWN,
and `UNKNOWN` is a real category studied on its own — never quietly folded into
inflow or outflow to reach a minimum sample size (B4.1.32).

The provider is optional and credential-driven. Chain-analytics APIs are
commercial; the core system runs without one and says so.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

from ..models import TransferContext
from ..providers import WhaleDataProvider, WhaleObservation
from .backoff import AttemptLog, ProviderFailure, RetryPolicy, call_with_retry, classify_http_status
from .evidence import RawEvidence, capture
from .policy import ProviderDeclaration, ProviderPolicy, RawRetention

WHALE_DECLARATION = ProviderDeclaration(
    provider_id="whales",
    policy=ProviderPolicy.AUTHENTICATED_LICENSED,
    purpose=(
        "Collect large on-chain BTC transfers with provider-supplied counterparty "
        "attribution, for the inflow/outflow/custody hypotheses B4 could not test."
    ),
    credentials_env="BTC_INTEL_WHALE_API_KEY",
    requires_paid_contract=True,
    rate_limit_note=(
        "Chain-analytics APIs meter by request and often by result volume. Configure "
        "minimum_interval_seconds from the contract."
    ),
    minimum_interval_seconds=1800,
    data_returned=(
        "observation id, chain, asset, amount, from/to attribution when the provider "
        "supplies it, provider classification and confidence, event timestamp"
    ),
    raw_retention=RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE,
    primary_source=False,
    official_source=False,
    terms_note=(
        "Attribution is the vendor's product and is generally not redistributable. "
        "Classification is never inferred locally: unattributed transfers stay UNKNOWN."
    ),
)

#: Provider vocabulary to the system's own. Anything unrecognised becomes
#: UNKNOWN rather than a best guess -- a wrong context is worse than no context,
#: because a study cannot tell it from a real one.
_CONTEXT_BY_NAME: dict[str, TransferContext] = {
    "EXCHANGE_INFLOW": TransferContext.EXCHANGE_INFLOW,
    "EXCHANGE_OUTFLOW": TransferContext.EXCHANGE_OUTFLOW,
    "CUSTODY": TransferContext.CUSTODY_TRANSFER,
    "CUSTODY_TRANSFER": TransferContext.CUSTODY_TRANSFER,
    "INTERNAL_TRANSFER": TransferContext.INTERNAL_EXCHANGE,
    "INTERNAL_EXCHANGE": TransferContext.INTERNAL_EXCHANGE,
}


def classify_transfer(
    provider_label: str | None,
    *,
    from_attribution: str | None,
    to_attribution: str | None,
) -> TransferContext:
    """Map a provider's own label to a transfer context.

    Attribution strings are carried through for the record but are *not* used to
    derive a classification: reading "binance-hot-wallet" out of a label and
    concluding "inflow" is exactly the inference this module refuses to make.
    Only the provider's explicit classification counts.
    """
    if not provider_label:
        return TransferContext.UNKNOWN
    return _CONTEXT_BY_NAME.get(provider_label.strip().upper(), TransferContext.UNKNOWN)


class DisabledWhaleProvider(WhaleDataProvider):
    """No whale contract configured. Reports the reason rather than empty results."""

    name = "whales"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.enabled = False

    def observations(self, start: datetime, end: datetime) -> list[WhaleObservation]:
        return []

    def status(self) -> dict[str, str]:
        return {"provider": self.name, "state": "DISABLED", "reason": self.reason}


class LicensedWhaleProvider(WhaleDataProvider):
    """Adapter for a contracted chain-analytics API."""

    name = "whales"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        provider_name: str = "whales",
        minimum_btc: float = 100.0,
        timeout: float = 15.0,
        opener: Callable[[str, str, float], bytes] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        retention: RawRetention = RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE,
    ) -> None:
        self.endpoint = endpoint
        self._api_key = api_key
        self.name = provider_name
        self.minimum_btc = minimum_btc
        self.timeout = timeout
        self.enabled = True
        self._opener = opener or _default_opener
        self._now = now
        self._retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        self._retention = retention
        self.last_evidence: list[RawEvidence] = []
        self.last_attempt = AttemptLog()

    def observations(self, start: datetime, end: datetime) -> list[WhaleObservation]:
        retrieved = self._now()
        self.last_evidence = []
        self.last_attempt = AttemptLog()

        query = urllib.parse.urlencode(
            {"start": start.isoformat(), "end": end.isoformat(), "min_btc": self.minimum_btc}
        )
        payload = call_with_retry(
            lambda: self._opener(f"{self.endpoint}?{query}", self._api_key, self.timeout),
            self._retry,
            sleep=self._sleep,
            log=self.last_attempt,
        )
        self.last_evidence.append(
            capture(
                self.name,
                payload,
                retention=self._retention,
                retrieved_at=retrieved,
                metadata={"min_btc": str(self.minimum_btc)},
            )
        )
        return parse_observations(json.loads(payload), retrieved, self.name)

    def status(self) -> dict[str, str]:
        return {"provider": self.name, "state": "ENABLED", "reason": ""}


def parse_observations(document: Any, retrieved: datetime, provider: str) -> list[WhaleObservation]:
    """Normalise a vendor response into observations.

    Availability is retrieval time. A transfer confirmed on-chain twenty minutes
    ago that this system learned of now became usable now -- the block timestamp
    is the event time, not the availability time, and conflating them would hand
    every whale study twenty minutes of hindsight.
    """
    observations: list[WhaleObservation] = []
    for record in document.get("transfers", []) if isinstance(document, dict) else []:
        try:
            observed = datetime.fromisoformat(str(record["observed_at"]).replace("Z", "+00:00"))
            context = classify_transfer(
                record.get("classification"),
                from_attribution=record.get("from_entity"),
                to_attribution=record.get("to_entity"),
            )
            observations.append(
                WhaleObservation(
                    observation_id=str(record["observation_id"]),
                    amount_btc=float(record["amount_btc"]),
                    from_entity=record.get("from_entity"),
                    to_entity=record.get("to_entity"),
                    transfer_context=context,
                    observed_at=observed,
                    available_at=retrieved,
                    source_url=str(record["source_url"]),
                    provider=provider,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(observations, key=lambda item: (item.available_at, item.observation_id))


def build_whale_provider(endpoint: str | None, api_key: str | None, **options: Any) -> WhaleDataProvider:
    """Return a working provider, or an honest disabled one."""
    if not endpoint:
        return DisabledWhaleProvider("no whale API endpoint is configured")
    if not api_key:
        return DisabledWhaleProvider(
            f"credential environment variable {WHALE_DECLARATION.credentials_env} is not set"
        )
    return LicensedWhaleProvider(endpoint, api_key, **options)


def context_counts(observations: list[WhaleObservation]) -> dict[str, int]:
    """B4.1.32. Counts per context, with every category always present.

    Every context appears even at zero, so a reader can tell "none observed"
    from "not tracked" -- and so no category can be quietly dropped from a
    report because it happened to be empty.
    """
    counts = {context.value: 0 for context in TransferContext}
    for observation in observations:
        counts[observation.transfer_context.value] += 1
    return counts


def _default_opener(url: str, api_key: str, timeout: float) -> bytes:  # pragma: no cover - needs a live API
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec: contracted endpoint
            return bytes(response.read())
    except urllib.error.HTTPError as error:
        raise ProviderFailure(classify_http_status(error.code), f"HTTP {error.code}") from error


__all__ = [
    "WHALE_DECLARATION",
    "DisabledWhaleProvider",
    "LicensedWhaleProvider",
    "build_whale_provider",
    "classify_transfer",
    "context_counts",
    "parse_observations",
]
