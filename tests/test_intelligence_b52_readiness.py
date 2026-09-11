"""B5.2 — a deterministic collection-readiness status, from Gate 1's own audit.

Once a collector runs for months, the question worth asking is how far the
corpus is from the gate it is collected for. `python -m market_intelligence.b5
status` answers it from the audit Gate 1 runs, under the committed
preregistration.

Pinned here:

* the instant is an argument, so two readings of one store are identical;
* every requirement shown is read from the preregistered policy object, and
  the module restates none of them;
* an empty store reads as NO_COLLECTION and insufficient;
* a collected store reports its collection record, funnel and sources, and
  still says insufficient -- elapsed time is not progress on its own;
* the store is opened read-only and left byte-identical;
* a store holding two extractor versions is refused until one is named.

No network.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.b5.__main__ import DEFAULT_PREREGISTRATION, EXIT_INTEGRITY, EXIT_OK, main
from market_intelligence.b5.contracts import Preregistration
from market_intelligence.collection import syndication
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.storage import IntelligenceStore
from tests.test_intelligence_deployment import FEED_PAYLOAD, minimal

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
AS_OF = "2026-09-05T00:00:00+00:00"


@pytest.fixture
def collected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Three scheduled cycles on one day, then nothing."""
    monkeypatch.setattr(syndication, "_default_opener", lambda url, timeout, agent=None: FEED_PAYLOAD)
    paths = StoragePaths.from_environment(tmp_path / "state").ensure()
    profile = CollectionProfile.from_mapping(minimal())
    for cycle in range(3):
        collect_once(paths, profile, now=lambda c=cycle: NOW + timedelta(hours=3 * c), require_free_bytes=1)
    return paths.database


def status(database: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, dict]:
    code = main(["status", "--db", str(database), "--as-of", AS_OF, "--json", *extra])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


class TestItIsTheGatesOwnReading:
    def test_two_readings_of_one_store_at_one_instant_are_identical(
        self, collected: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first, second = status(collected, capsys), status(collected, capsys)
        assert first == second and first[0] == EXIT_OK

    def test_every_requirement_is_the_preregistered_policys(
        self, collected: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload = status(collected, capsys)
        policy = Preregistration.read(DEFAULT_PREREGISTRATION).policy
        assert payload["required_events"] == policy.adequacy.minimum_events
        assert payload["required_effective_events"] == policy.adequacy.minimum_effective_events
        assert payload["required_publishers"] == policy.adequacy.minimum_publishers
        assert payload["required_span_days"] == policy.adequacy.minimum_span_days
        assert payload["required_coverage"] == policy.adequacy.minimum_coverage_fraction
        assert payload["required_ready_families"] == policy.minimum_ready_families
        assert payload["preregistration_hash"] == Preregistration.read(DEFAULT_PREREGISTRATION).content_hash()

    def test_the_module_restates_no_threshold(self) -> None:
        """Any number that is a threshold must come from the policy object, not a literal."""
        tree = ast.parse((REPO / "market_intelligence" / "b5" / "status.py").read_text(encoding="utf-8"))
        numbers = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)}
        assert numbers <= {0, 1}, f"threshold-like literals in status.py: {sorted(numbers)}"


class TestWhatItReports:
    def test_an_empty_store_is_no_collection_and_insufficient(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        database = tmp_path / "empty.duckdb"
        IntelligenceStore(database).close()
        code, payload = status(database, capsys)
        assert code == EXIT_OK
        assert payload["collection"]["state"] == "NO_COLLECTION" and payload["collection"]["coverage_fraction"] is None
        assert payload["gate1_passed"] is False and payload["decision"] == "INTELLIGENCE_CORPUS_INSUFFICIENT"

    def test_a_collected_store_reports_its_collection_and_is_still_insufficient(
        self, collected: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, payload = status(collected, capsys)
        collection = payload["collection"]
        assert (collection["successful_days"], collection["attempted_cycles"]) == (1, 3)
        assert collection["elapsed_days"] == 4 and collection["expected_cycles"] >= collection["attempted_cycles"]
        assert payload["documents"] >= 1 and payload["raw_events"] >= 1 and payload["publishers"] >= 1
        assert payload["gate1_passed"] is False and payload["ready_families"] == []
        assert payload["longest_family_span_days"] < payload["required_span_days"]

    def test_the_human_view_says_insufficient_and_names_the_span(
        self, collected: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["status", "--db", str(collected), "--as-of", AS_OF]) == EXIT_OK
        text = capsys.readouterr().out
        assert "The corpus is not sufficient." in text and "days required" in text
        assert "PASSED" not in text


class TestItTouchesNothing:
    def test_the_store_is_left_byte_identical(self, collected: Path, capsys: pytest.CaptureFixture[str]) -> None:
        before = hashlib.sha256(collected.read_bytes()).hexdigest()
        status(collected, capsys)
        assert hashlib.sha256(collected.read_bytes()).hexdigest() == before

    def test_an_instant_without_a_timezone_is_refused(self, collected: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["status", "--db", str(collected), "--as-of", "2026-09-05T00:00:00"]) == EXIT_INTEGRITY

    def test_a_store_with_two_extractor_versions_needs_one_named(
        self, collected: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = IntelligenceStore(collected)
        try:
            (event,) = store.signals_as_of(NOW + timedelta(days=1))[:1]
            store.put_signals([event.model_copy(update={"event_id": "legacy-" + event.event_id[:20], "extractor_version": "rules-v1"})])
        finally:
            store.close()
        assert main(["status", "--db", str(collected), "--as-of", AS_OF]) == EXIT_INTEGRITY
        capsys.readouterr()
        code, payload = status(collected, capsys, "--extractor-version", "rules-v2")
        assert code == EXIT_OK and payload["extractor_version"] == "rules-v2"
