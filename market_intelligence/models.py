from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


class SignalCategory(str, Enum):
    MACRO_MARKET = "MACRO_MARKET"
    CRYPTO_MARKET = "CRYPTO_MARKET"
    ONCHAIN = "ONCHAIN"
    WEB_EVENT = "WEB_EVENT"
    ENTITY_STATEMENT = "ENTITY_STATEMENT"


class EventType(str, Enum):
    REGULATION = "REGULATION"
    MONETARY_POLICY = "MONETARY_POLICY"
    ETF_FLOW = "ETF_FLOW"
    INSTITUTIONAL_ADOPTION = "INSTITUTIONAL_ADOPTION"
    EXCHANGE_INCIDENT = "EXCHANGE_INCIDENT"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"
    WHALE_TRANSFER = "WHALE_TRANSFER"
    MACRO_SHOCK = "MACRO_SHOCK"
    GEOPOLITICAL_EVENT = "GEOPOLITICAL_EVENT"
    LIQUIDITY_EVENT = "LIQUIDITY_EVENT"
    ENTITY_STATEMENT = "ENTITY_STATEMENT"


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class TransferContext(str, Enum):
    EXCHANGE_INFLOW = "EXCHANGE_INFLOW"
    EXCHANGE_OUTFLOW = "EXCHANGE_OUTFLOW"
    CUSTODY_TRANSFER = "CUSTODY_TRANSFER"
    INTERNAL_EXCHANGE = "INTERNAL_EXCHANGE"
    UNKNOWN = "UNKNOWN"


class ExtractionMethod(str, Enum):
    LLM = "LLM"
    RULE_BASED = "RULE_BASED"
    MANUAL = "MANUAL"
    FIXTURE = "FIXTURE"


class SourceType(str, Enum):
    PRIMARY_OFFICIAL = "PRIMARY_OFFICIAL"
    PRIMARY_CORPORATE = "PRIMARY_CORPORATE"
    REPUTABLE_NEWS = "REPUTABLE_NEWS"
    SPECIALIST_CRYPTO_NEWS = "SPECIALIST_CRYPTO_NEWS"
    SOCIAL_OR_STATEMENT = "SOCIAL_OR_STATEMENT"
    UNKNOWN = "UNKNOWN"


UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
SentimentScore = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]


class RetrievalProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str
    retrieved_at: datetime
    provider_document_id: str | None = None
    query: str | None = None
    query_id: str | None = None

    @field_validator("retrieved_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        return _utc(value)


class SourceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_type: SourceType = SourceType.UNKNOWN
    official_source: bool = False
    primary_source: bool = False
    known_publisher: bool = False
    timestamp_quality: UnitScore = 0.5
    content_completeness: UnitScore = 0.5


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    url: HttpUrl
    publisher: str
    title: str
    published_at: datetime | None = None
    retrieved_at: datetime
    available_at: datetime
    author: str | None = None
    text_hash: str
    query: str
    provider: str
    retrieval_provenance: tuple[RetrievalProvenance, ...] = ()
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata)
    schema_version: str = "document-v2"

    @field_validator("published_at", "retrieved_at", "available_at")
    @classmethod
    def timezone_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @field_validator("document_id", "publisher", "title", "text_hash", "query", "provider")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def availability_is_observable(self) -> "Document":
        # A retrieved document cannot have been known later than retrieval.
        if self.available_at > self.retrieved_at:
            raise ValueError("available_at cannot be after retrieved_at")
        return self

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def stable_id(url: str, text_hash: str) -> str:
        canonical = f"{url.strip()}\n{text_hash.lower()}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EventSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    event_time: datetime
    available_time: datetime
    source_ids: tuple[str, ...] = Field(min_length=1)
    category: SignalCategory
    entity: str | None = None
    event_type: EventType
    asset: str = "BTC"
    direction: Direction = Direction.UNKNOWN
    sentiment: SentimentScore
    btc_relevance: UnitScore
    novelty: UnitScore
    confidence: UnitScore
    expected_horizon_hours: Annotated[int, Field(ge=1, le=8760)]
    summary: str = Field(min_length=1, max_length=1000)
    transfer_context: TransferContext | None = None
    extractor_version: str
    extraction_method: ExtractionMethod = ExtractionMethod.FIXTURE
    schema_version: str = "event-v2"

    @field_validator("event_time", "available_time")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def enforce_semantics(self) -> "EventSignal":
        if self.event_type == EventType.WHALE_TRANSFER:
            if self.transfer_context is None:
                raise ValueError("whale transfer requires transfer_context")
            if self.transfer_context in {
                TransferContext.UNKNOWN,
                TransferContext.CUSTODY_TRANSFER,
                TransferContext.INTERNAL_EXCHANGE,
            } and self.direction != Direction.UNKNOWN:
                raise ValueError("ambiguous whale transfer direction must be UNKNOWN")
        elif self.transfer_context is not None:
            raise ValueError("transfer_context is only valid for whale transfers")
        return self

    @staticmethod
    def stable_id(source_ids: list[str] | tuple[str, ...], event_type: EventType, event_time: datetime) -> str:
        raw = "|".join(sorted(source_ids)) + f"|{event_type.value}|{_utc(event_time).isoformat()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EntityConfig(BaseModel):
    """Empirical prior/configuration only; importance is not asserted as fact."""

    name: str
    aliases: tuple[str, ...] = ()
    enabled: bool = True
    importance_weight: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    evaluation_notes: str | None = None
