"""B5.1 — collection runs rules-v2, and nothing counts two extractor versions at once.

Two properties:

* **What runs is rules-v2.** The deployed profile, the collection commands and
  every default that names an extractor version use the current one, and the
  profile hands the extractor each watched entity's declared event types.
* **Versions are never pooled.** Event identity carries the extractor version, so
  re-extracting a document under rules-v2 adds an event beside the rules-v1 one.
  B5's Gate 1 audit counts exactly one version: the only one present, or the one
  named -- and refuses a mixed store until one is named.

No network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.b5.__main__ import EXIT_INTEGRITY, EXIT_OK, main
from market_intelligence.b5.audit import audit_corpus, open_read_only, pinned_extractor_version
from market_intelligence.b5.contracts import default_preregistration
from market_intelligence.cli import build_parser
from market_intelligence.collection.fixtures import fixture_document
from market_intelligence.commands.collection import watchlist_extractor
from market_intelligence.configuration import WatchEntity
from market_intelligence.extractors import CURRENT_RULE_EXTRACTOR_VERSION, EvidenceRuleExtractor, RuleBasedExtractor
from market_intelligence.models import EventType
from market_intelligence.ops.profile import CollectionProfile
from market_intelligence.storage.store import IntelligenceStore

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "deploy" / "collection-profile.json"
AS_OF = datetime(2026, 9, 11, tzinfo=timezone.utc)


def profile() -> CollectionProfile:
    return CollectionProfile.load(PROFILE, environment={"BTC_INTEL_CONTACT": "operator@example.org"})


class TestWhatRunsIsRulesV2:
    def test_the_deployed_profile_extracts_with_rules_v2(self) -> None:
        extractor = profile().extractor()
        assert isinstance(extractor, EvidenceRuleExtractor)
        assert extractor.version == CURRENT_RULE_EXTRACTOR_VERSION == "rules-v2"

    def test_it_is_told_each_entitys_declared_event_types(self) -> None:
        raw = json.loads(PROFILE.read_text(encoding="utf-8"))
        declared = {entry["canonical_name"]: frozenset(EventType(v) for v in entry["expected_event_types"]) for entry in raw["watchlist"] if entry.get("expected_event_types")}
        assert profile().extractor().expected_types == declared

    def test_the_collection_commands_use_the_same_extractor(self) -> None:
        watchlist = [WatchEntity.model_validate(entry) for entry in json.loads(PROFILE.read_text(encoding="utf-8"))["watchlist"]]
        built = watchlist_extractor(watchlist)
        assert built.version == CURRENT_RULE_EXTRACTOR_VERSION
        assert built.expected_types == profile().extractor().expected_types

    def test_corpus_status_defaults_to_the_current_version(self) -> None:
        args = build_parser().parse_args(["--db", "x.duckdb", "corpus-status"])
        assert args.extractor_version == CURRENT_RULE_EXTRACTOR_VERSION

    def test_no_module_hard_codes_rules_v1_as_a_default(self) -> None:
        """rules-v1 is named once, where it is defined. Every default reads the current version."""
        offenders = [
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "market_intelligence").rglob("*.py"))
            if '"rules-v1"' in path.read_text(encoding="utf-8") and path.name != "extractors.py"
        ]
        assert offenders == []


def store_with(path: Path, *versions: str) -> Path:
    """One SEC document, extracted once per requested version."""
    retrieved = AS_OF - timedelta(days=2)
    doc = fixture_document(
        url="https://sec.example.gov/1",
        title="SEC Charges Fund Executives Under New Rule",
        retrieved_at=retrieved,
        publisher="U.S. Securities and Exchange Commission",
        published_at=retrieved - timedelta(hours=5),
        primary_source=True,
        official_source=True,
    )
    extractors = {"rules-v1": RuleBasedExtractor({"SEC": ()}), "rules-v2": EvidenceRuleExtractor({"SEC": ()}, {"SEC": (EventType.REGULATION,)})}
    store = IntelligenceStore(path)
    try:
        store.put_documents([doc])
        for version in versions:
            store.put_signals(extractors[version].extract([doc]))
    finally:
        store.close()
    return path


def audit(path: Path, version: str | None = None):
    connection = open_read_only(path)
    try:
        return audit_corpus(connection, as_of=AS_OF, extractor_version=version)
    finally:
        connection.close()


class TestVersionsAreNeverPooled:
    def test_a_single_version_is_pinned_automatically(self, tmp_path: Path) -> None:
        result = audit(store_with(tmp_path / "one.duckdb", "rules-v1"))
        assert result.audit.extractor_version == "rules-v1"
        assert result.audit.raw_events == 1

    def test_a_mixed_store_is_refused_until_one_version_is_named(self, tmp_path: Path) -> None:
        path = store_with(tmp_path / "mixed.duckdb", "rules-v1", "rules-v2")
        with pytest.raises(ValueError, match="pin one with --extractor-version"):
            audit(path)

    def test_a_named_version_counts_only_its_own_events(self, tmp_path: Path) -> None:
        path = store_with(tmp_path / "mixed.duckdb", "rules-v1", "rules-v2")
        v1, v2 = audit(path, "rules-v1"), audit(path, "rules-v2")
        assert (v1.audit.raw_events, v2.audit.raw_events) == (1, 1)
        assert v1.audit.extractor_versions_present == v2.audit.extractor_versions_present == ("rules-v1", "rules-v2")
        assert {r.extractor_version for r in v1.catalog} == {"rules-v1"} and {r.extractor_version for r in v2.catalog} == {"rules-v2"}
        # The same document: rules-v1's constant confidence is below the floor, rules-v2's is not.
        assert v1.audit.funnel["5_quality"] == 0 and v2.audit.funnel["5_quality"] == 1

    def test_an_empty_store_pins_nothing(self, tmp_path: Path) -> None:
        assert pinned_extractor_version([], None) is None
        assert audit(store_with(tmp_path / "empty.duckdb")).audit.extractor_version is None

    def test_the_command_refuses_a_mixed_store_and_accepts_a_pin(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = store_with(tmp_path / "mixed.duckdb", "rules-v1", "rules-v2")
        plan = str(default_preregistration().write(tmp_path / "plan.json"))
        base = ["audit", "--db", str(path), "--as-of", AS_OF.isoformat(), "--input-label", "fixture", "--preregistration", plan]
        assert main([*base, "--output", str(tmp_path / "refused")]) == EXIT_INTEGRITY
        assert "pin one with --extractor-version" in capsys.readouterr().err
        assert not (tmp_path / "refused" / "manifest.json").exists()
        assert main([*base, "--output", str(tmp_path / "v2"), "--extractor-version", "rules-v2"]) == EXIT_OK
        manifest = json.loads((tmp_path / "v2" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["input"]["extractor_version"] == "rules-v2"
