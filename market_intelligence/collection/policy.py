"""B4.1.1 / B4.1.41 — what a provider is allowed to be, and what it must declare.

Every production provider falls into exactly one lawfulness class, and the class
is declared in code rather than assumed from the provider's name. The point is
not bureaucracy: it is that "is this source lawful to poll continuously" is a
question with a real answer per source, and the answer belongs next to the
adapter that does the polling.

What is deliberately absent: any provider that scrapes a site whose terms or
technical controls prohibit it, bypasses a CAPTCHA, or circumvents a login. Not
because such a provider would be hard to write, but because a research corpus
whose provenance cannot be defended is worth less than no corpus at all — and
this whole track exists to build provenance.

A declaration also carries the operational facts an operator needs before
enabling anything: credentials, cadence, what comes back, and whether the raw
payload may be retained. `NON_REDISTRIBUTABLE_RAW_SOURCE` is a first-class
state, not an afterthought.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..errors import ConfigurationError


class ProviderPolicy(str, Enum):
    """B4.1.1. The four permitted classes."""

    #: A public feed or API the publisher documents and intends to be consumed.
    #: Official RSS/Atom, regulator feeds, documented open endpoints.
    PUBLIC_DOCUMENTED = "PUBLIC_DOCUMENTED"
    #: A commercial or contracted API used with credentials, under its terms.
    AUTHENTICATED_LICENSED = "AUTHENTICATED_LICENSED"
    #: A dataset the operator supplies themselves and vouches for.
    USER_SUPPLIED = "USER_SUPPLIED"
    #: Declared but not operable here — usually missing credentials. Reported,
    #: never silently skipped.
    DISABLED = "DISABLED"


class RawRetention(str, Enum):
    """B4.1.11 / B4.1.43. What may be kept of a provider's raw payload."""

    #: The payload may be stored and redistributed with the corpus.
    FULL = "FULL"
    #: The payload may be stored locally for reproduction but not redistributed.
    LOCAL_ONLY = "LOCAL_ONLY"
    #: Only a hash and permitted metadata may be kept.
    NON_REDISTRIBUTABLE_RAW_SOURCE = "NON_REDISTRIBUTABLE_RAW_SOURCE"


class ProviderDeclaration(BaseModel):
    """B4.1.41. Everything an operator needs before switching a provider on."""

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(min_length=1)
    policy: ProviderPolicy
    purpose: str = Field(min_length=1)
    #: Environment variable holding the credential, or None when none is needed.
    credentials_env: str | None = None
    #: Whether using this source at all requires a paid contract.
    requires_paid_contract: bool = False
    #: Operator-facing note on the provider's documented limits. Explicitly a
    #: snapshot: third-party terms and pricing change and this is not a promise.
    rate_limit_note: str = "no documented limit known; poll conservatively"
    #: Minimum seconds between calls this system will make, regardless of what
    #: the provider tolerates. Research need, not maximum extraction.
    minimum_interval_seconds: int = Field(default=900, ge=1)
    data_returned: str = Field(min_length=1)
    raw_retention: RawRetention = RawRetention.NON_REDISTRIBUTABLE_RAW_SOURCE
    #: True when the source is the party the news is *about*, not a report of it.
    primary_source: bool = False
    official_source: bool = False
    how_to_disable: str = "set enabled=false in the provider configuration"
    how_to_health_check: str = "btc-intel health"
    terms_note: str = ""

    def operable(self, credential_present: bool) -> bool:
        """Whether this provider can lawfully and practically run right now."""
        if self.policy is ProviderPolicy.DISABLED:
            return False
        if self.credentials_env and not credential_present:
            return False
        return True

    def disabled_reason(self, credential_present: bool) -> str | None:
        if self.policy is ProviderPolicy.DISABLED:
            return "declared DISABLED"
        if self.credentials_env and not credential_present:
            return f"credential environment variable {self.credentials_env} is not set"
        return None


class ProviderCatalogue:
    """The declarations, keyed by provider id.

    Registration refuses duplicates: two adapters claiming one id would make the
    lawfulness class of a collected document ambiguous, and the class is the
    thing the corpus's defensibility rests on.
    """

    def __init__(self, declarations: list[ProviderDeclaration] | None = None) -> None:
        self._declarations: dict[str, ProviderDeclaration] = {}
        for declaration in declarations or []:
            self.register(declaration)

    def register(self, declaration: ProviderDeclaration) -> None:
        if declaration.provider_id in self._declarations:
            raise ConfigurationError(f"provider already declared: {declaration.provider_id}")
        self._declarations[declaration.provider_id] = declaration

    def get(self, provider_id: str) -> ProviderDeclaration | None:
        return self._declarations.get(provider_id)

    def require(self, provider_id: str) -> ProviderDeclaration:
        declaration = self.get(provider_id)
        if declaration is None:
            raise ConfigurationError(
                f"provider {provider_id!r} has no policy declaration; every collecting provider "
                "must declare its lawfulness class before it may run"
            )
        return declaration

    def all(self) -> list[ProviderDeclaration]:
        return [self._declarations[key] for key in sorted(self._declarations)]

    def by_policy(self, policy: ProviderPolicy) -> list[ProviderDeclaration]:
        return [item for item in self.all() if item.policy is policy]


def redact(value: str | None) -> str:
    """B4.1.40. Render a secret safe for logs, manifests and API responses.

    Returns a fixed marker rather than a prefix or a length. A prefix leaks the
    key's issuer, and a length narrows a search — neither is worth the debugging
    convenience.
    """
    return "<redacted>" if value else "<unset>"


def redact_mapping(values: dict[str, str | None], secret_keys: tuple[str, ...] = ()) -> dict[str, str]:
    """Redact by key name as well as by explicit list.

    The name heuristic exists because the explicit list is the thing most likely
    to fall out of date when a new field is added.
    """
    suspicious = ("key", "token", "secret", "password", "authorization", "credential", "bearer")
    output: dict[str, str] = {}
    for key, value in values.items():
        lowered = key.casefold()
        if key in secret_keys or any(marker in lowered for marker in suspicious):
            output[key] = redact(value)
        else:
            output[key] = "" if value is None else str(value)
    return output


__all__ = [
    "ProviderCatalogue",
    "ProviderDeclaration",
    "ProviderPolicy",
    "RawRetention",
    "redact",
    "redact_mapping",
]
