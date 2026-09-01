"""B4.1.11 — raw evidence retention, hashed, with licensing respected.

Reproducing an extraction later needs the bytes the extractor actually saw. But
some providers' terms do not permit storing or redistributing their payloads,
and "we needed it for reproducibility" is not a licence.

So retention is per-provider policy, and the *hash is always kept* regardless.
A hash is derived data, not the content, and it is enough to prove that a
payload someone else holds is the one this system used — which recovers most of
the reproducibility value without redistributing anything.

The record is content-addressed and immutable. Re-storing identical bytes is a
no-op; storing different bytes under an existing id is an error, because that
would mean the evidence a stored extraction rests on had changed underneath it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ..errors import ReplayIntegrityError
from .policy import RawRetention


class RawEvidence(BaseModel):
    """One immutable raw payload record, or its hash alone."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    provider_id: str
    #: sha256 of the exact bytes the provider returned. Always present.
    content_hash: str
    content_bytes: int
    media_type: str = "application/json"
    retrieved_at: datetime
    retention: RawRetention
    #: The payload, when retention permits keeping it. None otherwise, and the
    #: distinction is explicit so a reader never mistakes "withheld" for "empty".
    payload: str | None = None
    #: Provider metadata that may be kept even when the payload may not.
    metadata: dict[str, str] = {}

    @field_validator("retrieved_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ReplayIntegrityError("raw evidence retrieved_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @property
    def redistributable(self) -> bool:
        return self.retention is RawRetention.FULL

    @property
    def payload_withheld(self) -> bool:
        return self.payload is None and self.retention is not RawRetention.FULL


def hash_payload(payload: bytes | str) -> str:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(raw).hexdigest()


def capture(
    provider_id: str,
    payload: bytes | str,
    *,
    retention: RawRetention,
    retrieved_at: datetime,
    media_type: str = "application/json",
    metadata: dict[str, str] | None = None,
) -> RawEvidence:
    """Build an evidence record, keeping the payload only when permitted.

    The evidence id is derived from provider and content hash, so the same bytes
    from the same provider always address the same record — which is what makes
    re-storing a no-op instead of a duplicate.
    """
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    content_hash = hash_payload(raw)
    keep = retention in (RawRetention.FULL, RawRetention.LOCAL_ONLY)
    return RawEvidence(
        evidence_id=hashlib.sha256(f"{provider_id}|{content_hash}".encode()).hexdigest()[:32],
        provider_id=provider_id,
        content_hash=content_hash,
        content_bytes=len(raw),
        media_type=media_type,
        retrieved_at=retrieved_at,
        retention=retention,
        payload=raw.decode("utf-8", errors="replace") if keep else None,
        metadata=dict(metadata or {}),
    )


class EvidenceStore:
    """Content-addressed, append-only raw evidence in DuckDB."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_evidence (
              evidence_id VARCHAR PRIMARY KEY,
              provider_id VARCHAR NOT NULL,
              content_hash VARCHAR NOT NULL,
              retrieved_at TIMESTAMPTZ NOT NULL,
              retention VARCHAR NOT NULL,
              payload JSON NOT NULL
            )
            """
        )

    def put(self, records: list[RawEvidence]) -> int:
        """Store new evidence. Identical re-storage is a no-op; a changed
        payload under an existing id is refused, because a stored extraction's
        evidence must not move underneath it."""
        stored = 0
        for record in records:
            existing = self.get(record.evidence_id)
            if existing is not None:
                if existing.content_hash != record.content_hash:
                    raise ReplayIntegrityError(
                        f"raw evidence {record.evidence_id} already exists with a different content hash"
                    )
                continue
            self.connection.execute(
                "INSERT INTO raw_evidence VALUES (?, ?, ?, ?, ?, ?)",
                [
                    record.evidence_id,
                    record.provider_id,
                    record.content_hash,
                    record.retrieved_at,
                    record.retention.value,
                    record.model_dump_json(),
                ],
            )
            stored += 1
        return stored

    def get(self, evidence_id: str) -> RawEvidence | None:
        row = self.connection.execute(
            "SELECT payload FROM raw_evidence WHERE evidence_id = ?", [evidence_id]
        ).fetchone()
        return RawEvidence.model_validate(json.loads(row[0])) if row else None

    def counts_by_retention(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT retention, count(*) FROM raw_evidence GROUP BY 1 ORDER BY 1"
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def total_bytes(self) -> int:
        """B4.1.42. Stored payload bytes, for growth measurement."""
        rows = self.connection.execute("SELECT payload FROM raw_evidence").fetchall()
        total = 0
        for (payload,) in rows:
            record = RawEvidence.model_validate(json.loads(payload))
            total += len(record.payload.encode("utf-8")) if record.payload else 0
        return total

    def export_redistributable(self, path: str | Path) -> dict[str, int]:
        """B4.1.44. Write only what may leave this machine, and say what did not."""
        rows = self.connection.execute("SELECT payload FROM raw_evidence ORDER BY evidence_id").fetchall()
        included: list[dict[str, Any]] = []
        omitted = 0
        for (payload,) in rows:
            record = RawEvidence.model_validate(json.loads(payload))
            if record.redistributable:
                included.append(record.model_dump(mode="json"))
            else:
                omitted += 1
                included.append(
                    {
                        "evidence_id": record.evidence_id,
                        "provider_id": record.provider_id,
                        "content_hash": record.content_hash,
                        "content_bytes": record.content_bytes,
                        "retrieved_at": record.retrieved_at.isoformat(),
                        "retention": record.retention.value,
                        "payload": None,
                        "omission_reason": "OMITTED_LICENSE",
                    }
                )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(json.dumps(included, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return {"records": len(included), "omitted_license": omitted}


__all__ = ["EvidenceStore", "RawEvidence", "capture", "hash_payload"]
