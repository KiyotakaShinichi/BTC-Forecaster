"""B5.1 — the operator's view states what collection actually did.

B5 found the collector had run for two days and was not deployed, and nothing an
operator reads would have said how much of the elapsed time was collected.
`ops-status` reported B4 readiness and a count of gap days computed only between
the first and last document -- a collector that stopped a week ago showed no gap
at all.

The B4.1-Ops machinery already exists and is not duplicated here: the scheduled
cycle and its exit codes, the run lock and stale-lock breaking, watermarks that
advance only on success, retries and failure isolation, the watchdog, backup,
restore, verification and the correction ledger all have their own tests. This
pins the one thing they did not report: the collection record, from the
canonical definition corpus-status and Gate 1 use, with the lag beside it.

* an empty deployment says NO_COLLECTION, never 0%, and claims nothing;
* scheduled cycles show up as collected days and cycles;
* a collector that has stopped shows how long ago it last succeeded, and the
  watchdog calls it stale -- software able to schedule itself is not a corpus
  that has been collected;
* nothing return-linked reaches the operations surface.

No network.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from market_intelligence.collection.coverage import CoverageState
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import CollectionProfile, collect_once
from market_intelligence.ops.summary import contains_forbidden_statistics
from market_intelligence.reports import ops_report
from market_intelligence.storage import IntelligenceStore
from tests.test_intelligence_silent_empty import NOW as SCHEDULED
from tests.test_intelligence_silent_empty import minimal, offline


def report_for(paths: StoragePaths) -> dict[str, Any]:
    store = IntelligenceStore(paths.database)
    try:
        return ops_report(store, paths.database, str(paths.root))
    finally:
        store.close()


def collected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cycles: int = 3) -> StoragePaths:
    """`cycles` scheduled cycles, three hours apart, on 2026-09-03 -- days before any run of this test."""
    offline(monkeypatch)
    paths = StoragePaths.from_environment(tmp_path / "s").ensure()
    profile = CollectionProfile.from_mapping(minimal())
    for cycle in range(cycles):
        collect_once(paths, profile, now=lambda c=cycle: SCHEDULED + timedelta(hours=3 * c), require_free_bytes=1)
    return paths


def test_an_empty_deployment_reports_no_collection_not_zero(tmp_path: Path) -> None:
    report = report_for(StoragePaths.from_environment(tmp_path / "s").ensure())
    assert report["collection"]["state"] == CoverageState.NO_COLLECTION.value
    assert report["collection"]["coverage_fraction"] is None
    assert report["collection_lag"]["documents"] == 0
    assert "not yet measurable" in report["human"]
    assert report["b4_readiness"] == "NOT_READY"


def test_scheduled_cycles_appear_as_collected_days_and_cycles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = report_for(collected(tmp_path, monkeypatch))
    collection = report["collection"]
    assert collection["state"] == CoverageState.MEASURED.value
    assert (collection["successful_days"], collection["attempted_cycles"], collection["successful_cycles"]) == (1, 3, 3)
    assert report["collection_lag"]["measured"] >= 1
    assert "elapsed day(s) with a successful provider attempt" in report["human"]
    assert "first seen - published" in report["human"]


def test_a_collector_that_stopped_is_reported_as_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three cycles on one day, then nothing: that is two days of software, not a corpus."""
    report = report_for(collected(tmp_path, monkeypatch))
    collection = report["collection"]
    assert collection["days_since_last_success"] >= 1
    assert collection["elapsed_days"] > collection["successful_days"]
    assert collection["coverage_fraction"] < 1.0
    assert any(alert["code"] == "COLLECTION_STALE" for alert in report["watchdog"]["alerts"])
    assert report["exit_code"] == 2
    assert report["b4_readiness"] == "NOT_READY"


def test_nothing_return_linked_reaches_the_operations_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert contains_forbidden_statistics(report_for(collected(tmp_path, monkeypatch, cycles=1))) == ()
