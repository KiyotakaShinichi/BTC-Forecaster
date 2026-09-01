"""B4.1.5 — SocialStatementProvider, lawful or off.

B4 could not test Trump, Musk, Powell or Saylor because no statement source
existed. This supplies the adapter — for a *documented, authenticated* API the
operator has a right to use. It does not supply a way around that requirement.

What is deliberately not here: scraping a social platform's web interface,
reading an unofficial mirror, or driving a logged-in session. Those would
produce statements, and would also produce a corpus whose provenance cannot be
defended and whose collection may breach the platform's terms. B4.1.1 rules them
out and this module does not quietly reintroduce them.

Without a configured credential the provider is `DISABLED` and *says so* — it
returns no statements and reports the reason. A disabled provider that looked
like an empty one would let "we collected nothing" masquerade as "nothing was
said", which is the same absence-versus-zero confusion that ended B4 on HOLD.

Availability is retrieval, as everywhere else. A statement posted three hours
before this system saw it became usable when this system saw it.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..providers import SocialStatement, SocialStatementProvider
from .backoff import AttemptLog, ProviderFailure, RetryPolicy, call_with_retry, classify_http_status
from .evidence import RawEvidence, capture
from .policy import ProviderDeclaration, ProviderPolicy, RawRetention

STATEMENT_DECLARATION = ProviderDeclaration(
    provider_id="statements",
    policy=ProviderPolicy.AUTHENTICATED_LICENSED,
    purpose=(
        "Collect dated public statements by watchlist entities through a documented, "
        "authenticated API the operator is licensed to use."
    ),
    credentials_env="BTC_INTEL_STATEMENTS_API_KEY",
    requires_paid_contract=True,
    rate_limit_note=(
        "Statement APIs are typically quota-metered per month as well as per minute. "
        "Set minimum_interval_seconds from the contract, not from research appetite."
    ),
    minimum_interval_seconds=3600,
    data_returned="statement id, author, permitted text representation, publication time, source URL",
    raw_retention=RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE,
    primary_source=True,
    official_source=False,
    terms_note=(
        "Most platforms forbid redistributing post content. Only a hash and permitted "
        "metadata leave this machine. No scraping, no login circumvention, no unofficial "
        "mirrors: without a licensed API this provider stays DISABLED."
    ),
)


class DisabledStatementProvider(SocialStatementProvider):
    """The honest no-op. Reports *why* it is off rather than returning silence."""

    name = "statements"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.enabled = False

    def statements(self, entities: list[str], start: datetime, end: datetime) -> list[SocialStatement]:
        return []

    def status(self) -> dict[str, str]:
        return {"provider": self.name, "state": "DISABLED", "reason": self.reason}


class LicensedStatementProvider(SocialStatementProvider):
    """Adapter for a documented statement API.

    The response contract is deliberately small and vendor-neutral: an id, an
    author, a permitted text representation, a publication timestamp and a source
    URL. Anything richer would bind the corpus to one vendor's schema, and the
    seam exists precisely so it is not.
    """

    name = "statements"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        provider_name: str = "statements",
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
        self.timeout = timeout
        self.enabled = True
        self._opener = opener or _default_opener
        self._now = now
        self._retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        self._retention = retention
        self.last_evidence: list[RawEvidence] = []
        self.last_attempt = AttemptLog()

    def statements(self, entities: list[str], start: datetime, end: datetime) -> list[SocialStatement]:
        retrieved = self._now()
        self.last_evidence = []
        self.last_attempt = AttemptLog()

        query = urllib.parse.urlencode(
            {"entities": ",".join(sorted(entities)), "start": start.isoformat(), "end": end.isoformat()}
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
                metadata={"entities": ",".join(sorted(entities))},
            )
        )

        document = json.loads(payload)
        return _parse_statements(document, retrieved, self.name)

    def status(self) -> dict[str, str]:
        return {"provider": self.name, "state": "ENABLED", "reason": ""}


def _parse_statements(document: Any, retrieved: datetime, provider: str) -> list[SocialStatement]:
    """Normalise a vendor response. A malformed record is skipped, not fatal."""
    statements: list[SocialStatement] = []
    for record in document.get("statements", []) if isinstance(document, dict) else []:
        try:
            published = datetime.fromisoformat(str(record["published_at"]).replace("Z", "+00:00"))
            statements.append(
                SocialStatement(
                    entity=str(record["entity"]),
                    statement_hash=str(record["statement_hash"]),
                    published_at=published,
                    # Availability is retrieval. A statement made hours ago that
                    # this system just received became usable just now.
                    available_at=retrieved,
                    source_url=str(record["source_url"]),
                    provider=provider,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(statements, key=lambda item: (item.available_at, item.statement_hash))


def build_statement_provider(
    endpoint: str | None,
    api_key: str | None,
    **options: Any,
) -> SocialStatementProvider:
    """Return a working provider, or an honest disabled one.

    Never raises for a missing credential: an operator running the core system
    without a statement contract is the expected case, and a crash would make
    the optional dependency mandatory in practice.
    """
    if not endpoint:
        return DisabledStatementProvider("no statement API endpoint is configured")
    if not api_key:
        return DisabledStatementProvider(
            f"credential environment variable {STATEMENT_DECLARATION.credentials_env} is not set"
        )
    return LicensedStatementProvider(endpoint, api_key, **options)


def statements_to_queries(entities: Sequence[str]) -> list[str]:
    """Entity names, sorted, for a deterministic request."""
    return sorted({entity.strip() for entity in entities if entity.strip()})


def _default_opener(url: str, api_key: str, timeout: float) -> bytes:  # pragma: no cover - needs a live API
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec: contracted endpoint
            return bytes(response.read())
    except urllib.error.HTTPError as error:
        raise ProviderFailure(classify_http_status(error.code), f"HTTP {error.code}") from error


__all__ = [
    "STATEMENT_DECLARATION",
    "DisabledStatementProvider",
    "LicensedStatementProvider",
    "build_statement_provider",
    "statements_to_queries",
]
