"""Bulk point-in-time replay (B3.1.2, B3.1.3).

The profile in ``research/b31/PROFILE.md`` established that the replay path is
``O(origins x history)`` and that the cost is Python-level, not database-level:
``SnapshotService.build_many`` already issues one bounded read per table, but
then re-scans the entire eligible history for every origin, eight times over, on
Pydantic objects.

This module replaces that inner loop. Three ideas do the work:

**Eligibility is a per-event property, not a per-origin test.** The reference
asks, for every origin, whether ``set(event.source_ids) <= available_document_ids``.
Because a document is available at ``origin`` exactly when
``document.available_at <= origin``, that condition is equivalent to::

    max(event.available_time, max(available_at of its source documents)) <= origin

Call that the event's *effective availability*. It is computed once per event.
An event whose source document is absent from the store is never eligible, and
gets an effective availability of ``+inf``.

**Window membership is a contiguous range.** Events arrive ordered by
``available_time``, so ``origin - W < available_time <= origin`` is a slice
locatable by two binary searches. And since effective availability is never
earlier than ``available_time``, ``effective <= origin`` already implies
``available_time <= origin`` -- so the upper edge needs no separate test.

**One pass, not eight.** The 24h window is a subset of the 72h window, so a
single ordered walk of the 72h slice accumulates every feature.

Bit-exactness
-------------
Feature floats are not merely compared against the reference -- they are hashed
into ``snapshot_id``. A last-bit difference is a different snapshot identity, so
the arithmetic has to match exactly, not approximately.

Two things make that harder than it looks, and the equivalence fixture caught
both:

1. **The reference associates the same product two different ways.**
   ``aggregate`` computes the 24h numerator as
   ``s.sentiment * s.btc_relevance * s.novelty * s.confidence``, which Python
   groups as ``((sent * rel) * nov) * conf``. ``_weighted_mean`` computes the
   72h numerator as ``s.sentiment * w`` with ``w = (rel * nov) * conf``. Those
   round differently, so this module keeps both columns.

2. **CPython's ``sum()`` is not a ``+=`` loop.** Since 3.12 its float fast path
   uses compensated (Neumaier) summation, which is *more* accurate than naive
   accumulation. Using ``+=`` changed ``sentiment_weighted_24h`` from ``0.14``
   to ``0.13999999999999993``. Rather than hand-roll Neumaier -- which would
   silently diverge if CPython changed its algorithm -- the single pass collects
   window values into lists and calls ``sum()``, so the summation semantics are
   whatever the interpreter's are.

That also rules out ``numpy.sum`` and prefix-sum differencing: both are faster
and both change the low bits.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .models import Document, EventSignal, EventType, TransferContext

#: Effective availability for an event whose source document is absent from the
#: store. The store's own integrity check rejects such events on write, so this
#: is a defensive guard for engines built directly from lists rather than a path
#: the persisted pipeline can reach.
NEVER_AVAILABLE = datetime.max.replace(tzinfo=timezone.utc)

#: Event types that contribute to ``macro_news_signal_72h``.
MACRO_EVENT_TYPES = frozenset({EventType.MACRO_SHOCK, EventType.MONETARY_POLICY, EventType.LIQUIDITY_EVENT})

#: Relevance threshold for ``high_relevance_event_count_24h``.
HIGH_RELEVANCE_THRESHOLD = 0.75

WINDOW_24H = timedelta(hours=24)
WINDOW_72H = timedelta(hours=72)


@dataclass(frozen=True)
class DocumentIndex:
    """Documents ordered by ``available_at``, with availability as a prefix.

    ``available_at`` is non-decreasing because the store reads documents with
    ``ORDER BY available_at``. The set of documents available at an origin is
    therefore always a prefix of this ordering, and its length is one binary
    search.
    """

    available_at: tuple[datetime, ...]
    document_ids: tuple[str, ...]
    text_hashes: tuple[str, ...]
    availability_by_id: dict[str, datetime]

    @classmethod
    def build(cls, documents: list[Document]) -> "DocumentIndex":
        ordered = sorted(documents, key=lambda document: document.available_at)
        availability: dict[str, datetime] = {}
        for document in ordered:
            # A document id appearing twice keeps its earliest availability;
            # that is when the evidence first existed.
            existing = availability.get(document.document_id)
            if existing is None or document.available_at < existing:
                availability[document.document_id] = document.available_at
        return cls(
            available_at=tuple(document.available_at for document in ordered),
            document_ids=tuple(document.document_id for document in ordered),
            text_hashes=tuple(document.text_hash for document in ordered),
            availability_by_id=availability,
        )

    def prefix_length(self, origin: datetime) -> int:
        """How many documents are available at ``origin``."""
        return bisect_right(self.available_at, origin)

    def __len__(self) -> int:
        return len(self.document_ids)


@dataclass(frozen=True)
class EventColumns:
    """Events as parallel plain-Python columns, ordered by ``available_time``.

    Extracting the columns once is the whole point: the reference re-reads
    Pydantic attributes for every event on every origin, and that is 52% of the
    measured wall time.

    Two weighted-sentiment columns, not one. The reference computes the 24h
    numerator as ``s.sentiment * s.btc_relevance * s.novelty * s.confidence``,
    which Python associates as ``((sent * rel) * nov) * conf``; but
    ``_weighted_mean`` computes the 72h numerator as ``s.sentiment * w`` where
    ``w = (rel * nov) * conf``. Those are different roundings of the same
    quantity. Collapsing them to one column changed the last bit of
    ``sentiment_weighted_24h`` and therefore the snapshot id -- caught by the
    bit-exact equivalence test, which is exactly what it is for.
    """

    available_time: tuple[datetime, ...]
    effective_time: tuple[datetime, ...]
    event_ids: tuple[str, ...]
    extractor_versions: tuple[str, ...]
    sentiment: tuple[float, ...]
    weight: tuple[float, ...]
    #: Two numerators, because the reference associates the product differently
    #: in the two places it computes one. See the class docstring.
    weighted_sentiment_24h: tuple[float, ...]
    weighted_sentiment_72h: tuple[float, ...]
    high_relevance: tuple[bool, ...]
    is_regulatory: tuple[bool, ...]
    is_exchange_inflow: tuple[bool, ...]
    is_macro: tuple[bool, ...]

    @classmethod
    def build(cls, events: list[EventSignal], documents: DocumentIndex) -> "EventColumns":
        ordered = sorted(events, key=lambda event: event.available_time)

        available: list[datetime] = []
        effective: list[datetime] = []
        event_ids: list[str] = []
        extractor_versions: list[str] = []
        sentiment: list[float] = []
        weight: list[float] = []
        weighted_24h: list[float] = []
        weighted_72h: list[float] = []
        high_relevance: list[bool] = []
        is_regulatory: list[bool] = []
        is_exchange_inflow: list[bool] = []
        is_macro: list[bool] = []

        for event in ordered:
            eligible_at = event.available_time
            unavailable = False
            for source_id in event.source_ids:
                source_available = documents.availability_by_id.get(source_id)
                if source_available is None:
                    # The source document is not in the store at all, so the
                    # subset test in the reference can never succeed.
                    unavailable = True
                    break
                if source_available > eligible_at:
                    eligible_at = source_available
            if unavailable:
                eligible_at = NEVER_AVAILABLE

            event_weight = event.btc_relevance * event.novelty * event.confidence
            available.append(event.available_time)
            effective.append(eligible_at)
            event_ids.append(event.event_id)
            extractor_versions.append(event.extractor_version)
            sentiment.append(event.sentiment)
            weight.append(event_weight)
            # Mirrors FeatureAggregator.aggregate, left-associated.
            weighted_24h.append(event.sentiment * event.btc_relevance * event.novelty * event.confidence)
            # Mirrors FeatureAggregator._weighted_mean.
            weighted_72h.append(event.sentiment * event_weight)
            high_relevance.append(event.btc_relevance >= HIGH_RELEVANCE_THRESHOLD)
            is_regulatory.append(event.event_type == EventType.REGULATION)
            is_exchange_inflow.append(
                event.event_type == EventType.WHALE_TRANSFER
                and event.transfer_context == TransferContext.EXCHANGE_INFLOW
            )
            is_macro.append(event.event_type in MACRO_EVENT_TYPES)

        return cls(
            available_time=tuple(available),
            effective_time=tuple(effective),
            event_ids=tuple(event_ids),
            extractor_versions=tuple(extractor_versions),
            sentiment=tuple(sentiment),
            weight=tuple(weight),
            weighted_sentiment_24h=tuple(weighted_24h),
            weighted_sentiment_72h=tuple(weighted_72h),
            high_relevance=tuple(high_relevance),
            is_regulatory=tuple(is_regulatory),
            is_exchange_inflow=tuple(is_exchange_inflow),
            is_macro=tuple(is_macro),
        )

    def __len__(self) -> int:
        return len(self.event_ids)


def _weighted_ratio(numerators: list[float], weights: list[float]) -> float:
    """Mirror of FeatureAggregator._weighted_mean, including its zero handling."""
    total = sum(weights)
    return sum(numerators) / total if total else 0.0


@dataclass(frozen=True)
class OriginEvidence:
    """Everything one origin needs, computed in a single ordered pass."""

    features: dict[str, float]
    eligible_event_ids: tuple[str, ...]
    eligible_extractor_versions: tuple[str, ...]
    document_prefix: int


class BulkReplayEngine:
    """Computes point-in-time evidence for many ordered origins.

    Correctness contract, asserted by the equivalence fixture: for every origin
    the features, the included document set and the included event set are
    identical to the reference implementation, bit-for-bit on the floats.
    """

    def __init__(self, documents: list[Document], events: list[EventSignal]) -> None:
        self.documents = DocumentIndex.build(documents)
        self.events = EventColumns.build(events, self.documents)

        # A second ordering, by effective availability, so the eligible set at
        # any origin is a prefix. See _eligible_membership.
        order = sorted(range(len(self.events)), key=lambda index: self.events.effective_time[index])
        self._effective_sorted: tuple[datetime, ...] = tuple(self.events.effective_time[i] for i in order)
        self._ids_by_effective: tuple[str, ...] = tuple(self.events.event_ids[i] for i in order)
        self._versions_by_effective: tuple[str, ...] = tuple(self.events.extractor_versions[i] for i in order)

    def evidence_for(self, origin: datetime) -> OriginEvidence:
        """Evidence available at ``origin``. Never reads past it."""
        columns = self.events

        # The 72h slice bounds the work. Effective availability is never earlier
        # than available_time, so an eligible event always has
        # available_time <= origin and lies at or before `upper`.
        upper = bisect_right(columns.available_time, origin)
        lower_72 = bisect_right(columns.available_time, origin - WINDOW_72H)
        lower_24 = bisect_right(columns.available_time, origin - WINDOW_24H)

        # Values are collected in reference visit order and summed with the
        # builtin sum(), which is compensated -- see the module docstring.
        sentiment_24: list[float] = []
        weight_24: list[float] = []
        weighted_24: list[float] = []
        high_relevance_24 = 0

        regulatory_weight: list[float] = []
        regulatory_weighted: list[float] = []
        inflow_weight: list[float] = []
        inflow_weighted: list[float] = []
        macro_weight: list[float] = []
        macro_weighted: list[float] = []

        for index in range(lower_72, upper):
            if columns.effective_time[index] > origin:
                continue

            # Independent tests, not elif: the reference builds three separate
            # filtered lists. The categories are disjoint today, and chaining
            # them would silently depend on that staying true.
            if columns.is_regulatory[index]:
                regulatory_weight.append(columns.weight[index])
                regulatory_weighted.append(columns.weighted_sentiment_72h[index])
            if columns.is_exchange_inflow[index]:
                inflow_weight.append(columns.weight[index])
                inflow_weighted.append(columns.weighted_sentiment_72h[index])
            if columns.is_macro[index]:
                macro_weight.append(columns.weight[index])
                macro_weighted.append(columns.weighted_sentiment_72h[index])

            if index >= lower_24:
                sentiment_24.append(columns.sentiment[index])
                weight_24.append(columns.weight[index])
                weighted_24.append(columns.weighted_sentiment_24h[index])
                if columns.high_relevance[index]:
                    high_relevance_24 += 1

        count_24 = len(sentiment_24)
        total_weight_24 = sum(weight_24)
        features = {
            "sentiment_mean_24h": sum(sentiment_24) / count_24 if count_24 else 0.0,
            "sentiment_weighted_24h": sum(weighted_24) / total_weight_24 if total_weight_24 else 0.0,
            "event_count_24h": float(count_24),
            "high_relevance_event_count_24h": float(high_relevance_24),
            "regulatory_signal_72h": _weighted_ratio(regulatory_weighted, regulatory_weight),
            "whale_exchange_inflow_signal_72h": _weighted_ratio(inflow_weighted, inflow_weight),
            "macro_news_signal_72h": _weighted_ratio(macro_weighted, macro_weight),
        }

        eligible_event_ids, eligible_extractor_versions = self._eligible_membership(origin)
        return OriginEvidence(
            features=features,
            eligible_event_ids=eligible_event_ids,
            eligible_extractor_versions=eligible_extractor_versions,
            document_prefix=self.documents.prefix_length(origin),
        )

    def _eligible_membership(self, origin: datetime) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Every eligible event at ``origin``, not only the windowed ones.

        Eligibility is monotone: once an event's effective availability has
        passed, it never becomes ineligible again. Walking the whole history per
        origin was therefore O(origins x history) even after the window work was
        made cheap -- it was 1.4 million list appends across 200 origins. Sorting
        the events once by effective availability turns it into a binary search
        for the prefix length.

        Snapshot identity sorts both lists, so the order here is irrelevant.
        """
        cutoff = bisect_right(self._effective_sorted, origin)
        # Tuple slices, not list copies: IntelligenceSnapshot.create sorts both
        # immediately, so a second copy per origin buys nothing.
        return self._ids_by_effective[:cutoff], self._versions_by_effective[:cutoff]

    def document_membership(self, prefix: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Document ids and source hashes available at a prefix length."""
        return self.documents.document_ids[:prefix], self.documents.text_hashes[:prefix]


__all__ = [
    "HIGH_RELEVANCE_THRESHOLD",
    "MACRO_EVENT_TYPES",
    "NEVER_AVAILABLE",
    "WINDOW_24H",
    "WINDOW_72H",
    "BulkReplayEngine",
    "DocumentIndex",
    "EventColumns",
    "OriginEvidence",
]
