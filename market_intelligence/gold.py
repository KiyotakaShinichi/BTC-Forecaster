from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import GoldEvaluationReport, GoldLabel, GoldManifest, build_gold_report
from .extractors import EventExtractor
from .models import Document


def load_gold_set(directory: str | Path | None = None) -> tuple[list[Document], list[GoldLabel], GoldManifest]:
    root = Path(directory) if directory else Path(__file__).parent
    fixtures_path = root / "gold_fixtures.json"
    documents_path = root / "gold_documents.json"
    manifest = GoldManifest.model_validate_json((root / "gold_manifest.json").read_text(encoding="utf-8"))
    fixture_bytes = fixtures_path.read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != manifest.case_hash:
        raise ValueError("gold fixture hash mismatch")
    raw_labels = json.loads(fixture_bytes)
    raw_documents = json.loads(documents_path.read_text(encoding="utf-8"))
    available = datetime(2025, 1, 1, tzinfo=timezone.utc)
    documents = []
    for item in raw_documents:
        text_hash = Document.content_hash(item["title"])
        documents.append(
            Document(
                document_id=item["document_id"],
                url=f"https://example.invalid/gold/{item['document_id']}",
                publisher="Curated Gold Fixture",
                title=item["title"],
                published_at=available,
                retrieved_at=available,
                available_at=available,
                text_hash=text_hash,
                query="gold evaluation",
                provider="gold-fixture",
            )
        )
    return documents, [GoldLabel.model_validate(item) for item in raw_labels], manifest


def write_gold_evaluation(extractor: EventExtractor, output_path: str | Path) -> GoldEvaluationReport:
    documents, labels, manifest = load_gold_set()
    report = build_gold_report(extractor, documents, labels, manifest)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return report
