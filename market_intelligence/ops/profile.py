"""A deployable collection profile: one file that fully describes what to collect.

`run_scheduled` takes callables. That is right for a library -- it lets a test
substitute anything -- but a scheduler cannot invoke a callable. Something has to
turn durable, reviewable configuration into those callables, and that translation
is where a deployment quietly goes wrong: a feed silently dropped, a watchlist
edited on the host and never committed, an authenticated provider that looks
enabled because a variable name was spelled correctly in a config file while the
credential behind it was never set.

So the profile is deliberately narrow:

* **Feeds are selected by id from the committed catalogue**, never by URL. An
  operator cannot point this at an arbitrary endpoint by editing JSON on the
  host; adding a source is a reviewed change to `feeds.py`. The allowlist stays
  the policy (B4.1.1).
* **An authenticated provider is enabled only when its credential is actually
  present in the environment.** Naming it is not enabling it. A provider that is
  configured but has no credential is reported as unavailable, with the name of
  the variable to set -- never guessed at, never defaulted, never logged.
* **The fingerprint that lands in the manifest carries no secrets.** It records
  which providers ran and which feeds were configured, so a later replay can
  tell "this feed was not configured yet" from "this feed returned nothing".
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..collection.backoff import RetryPolicy
from ..collection.feeds import NEWS_API_DECLARATION, SYNDICATION_DECLARATION, feeds_by_id
from ..collection.policy import ProviderDeclaration
from ..collection.syndication import FeedSource, SyndicationProvider
from ..configuration import (
    ProviderCategory,
    ProviderConfig,
    QueryPlanner,
    QuerySpec,
    WatchEntity,
    configuration_fingerprint,
)
from ..errors import ConfigurationError
from ..extractors import RuleBasedExtractor
from ..retrieval import MultiProviderRetriever
from .paths import StoragePaths
from .scheduled import ScheduledOutcome, run_scheduled

SYNDICATION = "syndication"


#: The one variable a collection deployment genuinely requires. Named as a
#: constant rather than written inline because it is read through an aliased
#: mapping -- `env.get(...)`, where `env` may be a test double -- which no scan
#: of `os.environ` can see. `.env.example` is checked against declarations like
#: this one, so an undeclared read is an undocumented variable.
CONTACT_ENV = "BTC_INTEL_CONTACT"

#: Substituted into `user_agent` so a contact address never has to be committed.
#: The address identifies a person, and a repository is the wrong place to
#: publish one; the host supplies it and only the shape lives in the profile.
CONTACT_PLACEHOLDER = "${" + CONTACT_ENV + "}"


def _resolve_user_agent(
    template: str | None, origin: str, environment: Mapping[str, str] | None
) -> str | None:
    """Expand the contact placeholder, and refuse anything a publisher cannot use.

    Three ways this goes wrong, all of them caught here rather than weeks into a
    deployment when a feed starts answering 403:

    * The placeholder is left unexpanded, and every request advertises a literal
      `${BTC_INTEL_CONTACT}` -- worse than no contact, because it looks like one.
    * A bare product string with no contact at all, which gives a publisher who
      wants to complain nowhere to complain to.
    * The variable is set to something empty, which silently collapses to the
      second case.
    """
    if template is None:
        return None
    env = os.environ if environment is None else environment

    if CONTACT_PLACEHOLDER in template:
        contact = (env.get(CONTACT_ENV) or "").strip()
        if not contact:
            raise ConfigurationError(
                f"collection profile {origin} asks for a contact address via "
                "BTC_INTEL_CONTACT, which is not set. Set it to an address a publisher "
                "can write to, or remove the user_agent field and accept the default. "
                "Do not leave the placeholder unexpanded: a request advertising a "
                "literal ${BTC_INTEL_CONTACT} is worse than one advertising nothing."
            )
        template = template.replace(CONTACT_PLACEHOLDER, contact)

    if "@" not in template and "http" not in template:
        raise ConfigurationError(
            f"collection profile {origin} sets a user_agent with no contact address. "
            "Publishers that block a collector need somewhere to write first; give an "
            "email or a URL, or leave the field out and accept the default."
        )
    return template


@dataclass(frozen=True)
class ProviderAvailability:
    """Whether a provider can genuinely run, and what is missing if it cannot."""

    provider_id: str
    configured: bool
    available: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "configured": self.configured,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CollectionProfile:
    """What a deployed collector watches, where it looks, and how often."""

    name: str
    entities: tuple[WatchEntity, ...]
    feeds: tuple[FeedSource, ...]
    minimum_interval_seconds: int = 900
    timeout: float = 20.0
    user_agent: str | None = None
    max_attempts: int = 3

    @classmethod
    def load(
        cls, path: str | Path, *, environment: Mapping[str, str] | None = None
    ) -> "CollectionProfile":
        """Read a profile, failing loudly on anything ambiguous."""
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ConfigurationError(f"collection profile not found: {source}") from error
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"collection profile {source} is not valid JSON: {error}") from error
        if not isinstance(raw, dict):
            raise ConfigurationError(f"collection profile {source} must be a JSON object")
        return cls.from_mapping(raw, origin=str(source), environment=environment)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        origin: str = "<mapping>",
        environment: Mapping[str, str] | None = None,
    ) -> "CollectionProfile":
        missing = [key for key in ("name", "watchlist", "feeds") if key not in raw]
        if missing:
            raise ConfigurationError(f"collection profile {origin} is missing: {', '.join(missing)}")

        entities = tuple(WatchEntity.model_validate(entry) for entry in raw["watchlist"])
        enabled = tuple(entity for entity in entities if entity.enabled)
        if not enabled:
            raise ConfigurationError(f"collection profile {origin} has no enabled watch entities")

        feed_ids = tuple(str(feed_id) for feed_id in raw["feeds"])
        if not feed_ids:
            raise ConfigurationError(f"collection profile {origin} enables no feeds")
        try:
            feeds = feeds_by_id(*feed_ids)
        except KeyError as error:
            raise ConfigurationError(
                f"collection profile {origin} names feeds that are not in the committed "
                f"catalogue ({error}). Feeds are added by a reviewed change to feeds.py, "
                "not by editing configuration on the host."
            ) from error

        interval = int(raw.get("minimum_interval_seconds", 900))
        declared = SYNDICATION_DECLARATION.minimum_interval_seconds
        if interval < declared:
            raise ConfigurationError(
                f"collection profile {origin} asks to poll every {interval}s, faster than the "
                f"{declared}s the syndication provider declares. Change the declaration, and "
                "argue for it, rather than overriding it here."
            )

        user_agent = _resolve_user_agent(raw.get("user_agent") or None, origin, environment)

        return cls(
            name=str(raw["name"]),
            entities=enabled,
            feeds=feeds,
            minimum_interval_seconds=interval,
            timeout=float(raw.get("timeout_seconds", 20.0)),
            user_agent=user_agent,
            max_attempts=int(raw.get("max_attempts", 3)),
        )

    # ------------------------------------------------------------------ wiring

    def aliases(self) -> dict[str, tuple[str, ...]]:
        return {entity.canonical_name: entity.aliases for entity in self.entities}

    def extractor(self) -> RuleBasedExtractor:
        return RuleBasedExtractor(self.aliases())

    def build_queries(self, moment: datetime) -> list[QuerySpec]:
        return QueryPlanner().plan(list(self.entities), moment)

    def declarations(self) -> dict[str, ProviderDeclaration]:
        """Only providers that can actually run. Declaring is not enabling."""
        declaration = SYNDICATION_DECLARATION
        if self.minimum_interval_seconds != declaration.minimum_interval_seconds:
            declaration = declaration.model_copy(
                update={"minimum_interval_seconds": self.minimum_interval_seconds}
            )
        return {SYNDICATION: declaration}

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            id=SYNDICATION,
            type="syndication",
            source_category=ProviderCategory.OFFICIAL_GOVERNMENT,
            timeout=self.timeout,
        )

    def build_retriever(
        self,
        due: Sequence[str],
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> MultiProviderRetriever:
        """Build only what is due. A provider that is not due is never constructed.

        The clock is passed in rather than defaulted per provider: the provider
        stamps `available_at` and the cycle stamps everything else, and two
        clocks in one cycle is one clock too many.
        """
        if SYNDICATION not in due:
            return MultiProviderRetriever({}, {})
        provider = SyndicationProvider(
            self.feeds,
            timeout=self.timeout,
            retry=RetryPolicy(max_attempts=self.max_attempts),
            user_agent=self.user_agent,
            now=now,
        )
        return MultiProviderRetriever({SYNDICATION: provider}, {SYNDICATION: self.provider_config()})

    # ----------------------------------------------------------- reportability

    def availability(
        self, environment: Mapping[str, str] | None = None
    ) -> tuple[ProviderAvailability, ...]:
        """What can run right now, and what an operator must supply for the rest.

        Reported for the licensed provider too, not only the enabled one: an
        operator needs to see that a broader source exists and is switched off,
        otherwise "we only have government feeds" reads as a design decision
        rather than as a missing contract.
        """
        env = os.environ if environment is None else environment
        variable = NEWS_API_DECLARATION.credentials_env
        has_credential = bool(variable and env.get(variable))
        return (
            ProviderAvailability(
                provider_id=SYNDICATION,
                configured=True,
                available=True,
                reason=(
                    f"{len(self.feeds)} public feeds, no credentials required"
                    if self.user_agent
                    else f"{len(self.feeds)} public feeds, no credentials required; no contact "
                    "User-Agent configured, which some government feeds answer 403 to"
                ),
            ),
            ProviderAvailability(
                provider_id=NEWS_API_DECLARATION.provider_id,
                configured=False,
                available=False,
                reason=(
                    f"{variable} is set, but this profile does not enable the provider"
                    if has_credential
                    else f"not enabled: requires a paid contract and {variable}"
                ),
            ),
        )

    def fingerprint(self) -> dict[str, Any]:
        """Configuration as recorded in the manifest. Never carries a secret."""
        return {
            "profile": self.name,
            "entities": sorted(entity.canonical_name for entity in self.entities),
            "feeds": sorted(feed.feed_id for feed in self.feeds),
            "providers": [SYNDICATION],
            "minimum_interval_seconds": self.minimum_interval_seconds,
            "contact_user_agent_configured": self.user_agent is not None,
        }


def record_configuration(paths: StoragePaths, profile: CollectionProfile) -> Path:
    """Write the preimage of the manifest's `configuration_fingerprint`.

    Every manifest carries a hash of the configuration that produced it, which
    is enough to prove two runs were configured identically and not enough to
    say what either was configured *with*. Six months on, that is the difference
    between "the Treasury feed was quiet" and "the Treasury feed was not
    switched on yet" -- a coverage gap that looks exactly like an absence of
    news unless the preimage was kept.

    Content-addressed and first-write-wins, so a stable configuration writes one
    small file however many cycles run, and a changed one cannot overwrite the
    record of what came before it.
    """
    fingerprint = configuration_fingerprint(profile.fingerprint())
    target = paths.manifests / f"configuration-{fingerprint}.json"
    if not target.exists():
        paths.manifests.mkdir(parents=True, exist_ok=True)
        payload = {"configuration_fingerprint": fingerprint, "configuration": profile.fingerprint()}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def collect_once(
    paths: StoragePaths,
    profile: CollectionProfile,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    require_free_bytes: int = 64 * 1024 * 1024,
    source_sha: str | None = None,
) -> ScheduledOutcome:
    """One scheduled collection cycle driven entirely by a committed profile."""
    paths.ensure()
    record_configuration(paths, profile)
    return run_scheduled(
        paths,
        build_queries=profile.build_queries,
        build_retriever=lambda due: profile.build_retriever(due, now=now),
        extractor=profile.extractor(),
        configuration=profile.fingerprint(),
        declarations=profile.declarations(),
        now=now,
        require_free_bytes=require_free_bytes,
        source_sha=source_sha,
    )


__all__ = ["CollectionProfile", "ProviderAvailability", "collect_once", "record_configuration"]
