from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ConfigurationError
from .models import EventType


class ProviderCategory(str, Enum):
    GENERAL_WEB = "GENERAL_WEB"
    NEWS = "NEWS"
    OFFICIAL_GOVERNMENT = "OFFICIAL_GOVERNMENT"
    CENTRAL_BANK = "CENTRAL_BANK"
    REGULATORY = "REGULATORY"
    CRYPTO_NEWS = "CRYPTO_NEWS"
    ONCHAIN = "ONCHAIN"
    SOCIAL = "SOCIAL"
    MARKET_DATA = "MARKET_DATA"


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    enabled: bool = True
    priority: Annotated[int, Field(ge=0, le=100)] = 50
    timeout: Annotated[float, Field(gt=0, le=300)] = 15.0
    rate_limit: Annotated[float | None, Field(gt=0)] = None
    credentials_env: str | None = None
    source_category: ProviderCategory
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("credentials_env")
    @classmethod
    def valid_env_name(cls, value: str | None) -> str | None:
        if value and (not value.replace("_", "A").isalnum() or value[0].isdigit()):
            raise ValueError("credentials_env must be an environment variable name")
        return value

    def credential(self) -> str | None:
        return os.environ.get(self.credentials_env) if self.credentials_env else None


class ProviderRegistry:
    def __init__(self, factories: dict[str, Any] | None = None):
        self._factories = dict(factories or {})

    def register(self, provider_type: str, factory: Any) -> None:
        if provider_type in self._factories:
            raise ConfigurationError(f"provider type already registered: {provider_type}")
        self._factories[provider_type] = factory

    def build(self, configs: list[ProviderConfig]) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for config in sorted(configs, key=lambda c: (c.priority, c.id)):
            if not config.enabled:
                continue
            factory = self._factories.get(config.type)
            if factory is None:
                raise ConfigurationError(f"unknown provider type: {config.type}")
            if config.credentials_env and config.credential() is None:
                raise ConfigurationError(f"credential environment variable is not set: {config.credentials_env}")
            providers[config.id] = factory(config)
        return providers


class EntityType(str, Enum):
    POLITICAL = "POLITICAL"
    CENTRAL_BANK = "CENTRAL_BANK"
    REGULATOR = "REGULATOR"
    CORPORATE = "CORPORATE"
    CRYPTO_EXECUTIVE = "CRYPTO_EXECUTIVE"
    INSTITUTION = "INSTITUTION"
    EXCHANGE = "EXCHANGE"
    ETF_ISSUER = "ETF_ISSUER"


class WatchEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    canonical_name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    entity_type: EntityType
    enabled: bool = True
    topics: tuple[str, ...] = Field(min_length=1)
    expected_event_types: tuple[EventType, ...] = ()
    importance_prior: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    measured_impact: float | None = None


class QuerySpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_id: str
    query: str
    topic: str
    entities: tuple[str, ...]
    event_types: tuple[EventType, ...]
    lookback_hours: Annotated[int, Field(ge=1, le=24 * 90)]
    priority: Annotated[int, Field(ge=0, le=100)]
    generated_at: datetime


class QueryPlanner:
    version = "query-planner-v1"

    def plan(self, watchlist: list[WatchEntity], generated_at: datetime, lookback_hours: int = 24) -> list[QuerySpec]:
        if generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        generated_at = generated_at.astimezone(timezone.utc)
        queries: list[QuerySpec] = []
        for entity in sorted((e for e in watchlist if e.enabled), key=lambda e: e.canonical_name.casefold()):
            for topic in sorted(set(entity.topics), key=str.casefold):
                text = f'"{entity.canonical_name}" {topic}'
                material = json.dumps(
                    {
                        "entity": entity.canonical_name,
                        "topic": topic,
                        "lookback_hours": lookback_hours,
                        "version": self.version,
                    },
                    sort_keys=True,
                )
                queries.append(
                    QuerySpec(
                        query_id=hashlib.sha256(material.encode()).hexdigest(),
                        query=text,
                        topic=topic,
                        entities=(entity.canonical_name,),
                        event_types=entity.expected_event_types,
                        lookback_hours=lookback_hours,
                        priority=50,
                        generated_at=generated_at,
                    )
                )
        return queries


def configuration_fingerprint(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: item.model_dump(mode="json") if isinstance(item, BaseModel) else str(item),
        ).encode()
    ).hexdigest()
