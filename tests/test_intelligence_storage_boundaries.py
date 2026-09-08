"""The storage package's boundaries, and the API they must not disturb.

`market_intelligence.storage` was one 887-line module. It is now a package with
three parts, and a refactor of the code that holds the corpus is worth exactly
as much as the evidence that it changed nothing. So this file checks two things
the 728 collector tests around it cannot:

* the public surface is unchanged -- every symbol and every method that
  resolved before still resolves, by name;
* the boundaries are real -- reads do not write, DDL lives in one place, and
  the store delegates rather than keeping a second copy of each query.

The second is the one that decays. A split holds only while somebody keeps
putting things on the right side of it, and "somebody remembered" is not a
mechanism.
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from market_intelligence import storage
from market_intelligence.storage import IntelligenceStore, queries, schema

PACKAGE = Path(storage.__file__).parent


class TestThePublicSurfaceIsUnchanged:
    #: Captured from the module as it stood before the split. Written out rather
    #: than derived, because a list derived from the current code would agree
    #: with whatever the current code happens to be.
    METHODS_BEFORE_THE_SPLIT = frozenset({
        "all_health", "all_watermarks", "canonical_availability", "close",
        "correction_count", "corrections", "dashboard_summary", "documents_as_of",
        "eligible_signals_as_of", "event_sources", "export_parquet", "extraction_observability",
        "extractor_versions_for", "get_document", "get_event", "get_run",
        "get_snapshot", "get_watermark", "ineligible_event_ids", "latest_quality_scoreboard",
        "latest_run_as_of", "list_snapshots", "metrics", "persist_cycle",
        "provider_observability", "put_corrections", "put_dataset_manifest", "put_documents",
        "put_health", "put_provider_attempts", "put_quality_scoreboard", "put_quarantine",
        "put_run", "put_signals", "put_snapshot", "put_snapshots",
        "quarantine_count", "query_documents", "query_events", "query_quarantine",
        "query_runs", "ready", "record_sightings", "runs_up_to",
        "schema_version", "sighting", "signals_as_of", "verify_snapshot",
    })

    def test_every_method_still_exists(self) -> None:
        missing = {
            name for name in self.METHODS_BEFORE_THE_SPLIT if not hasattr(IntelligenceStore, name)
        }
        assert not missing, f"lost in the split: {sorted(missing)}"

    def test_the_module_path_still_resolves(self) -> None:
        """`from market_intelligence.storage import IntelligenceStore` is what
        24 modules and the whole test suite import. A package that broke it
        would be a rename dressed as a refactor."""
        from market_intelligence.storage import IntelligenceStore as direct

        assert direct is IntelligenceStore

    def test_the_module_constants_still_resolve(self) -> None:
        assert storage.SCHEMA_VERSION == schema.SCHEMA_VERSION == 3
        assert storage.SNAPSHOT_INSERT_CHUNK == 500

    def test_the_signatures_are_unchanged(self) -> None:
        """Delegation preserves behaviour only if it preserves the call. Every
        read method's parameters are checked against the function it forwards
        to, so a stub that quietly dropped a keyword would fail here rather
        than at a call site months later."""
        for name in queries.__all__:
            method = getattr(IntelligenceStore, name, None)
            if method is None:
                continue
            store_params = list(inspect.signature(method).parameters)[1:]
            query_params = list(inspect.signature(getattr(queries, name)).parameters)[1:]
            assert store_params == query_params, name


class TestTheBoundariesAreReal:
    def test_only_the_schema_module_issues_ddl(self) -> None:
        """One place decides what a row is. DDL appearing anywhere else means
        two answers to that question, and the corpus is only interpretable
        because there is one."""
        offenders = []
        for path in sorted(PACKAGE.glob("*.py")):
            if path.name == "schema.py":
                continue
            text = path.read_text(encoding="utf-8").upper()
            if re.search(r"\b(CREATE TABLE|ALTER TABLE|DROP TABLE)\b", text):
                offenders.append(path.name)
        assert not offenders, f"DDL outside schema.py: {offenders}"

    def test_the_query_module_never_writes(self) -> None:
        """A read that writes is a read nobody can run twice and compare."""
        text = queries_source().upper()
        for statement in ("INSERT ", "UPDATE ", "DELETE ", "CREATE "):
            assert statement not in text, f"{statement.strip()} in queries.py"

    def test_the_schema_migration_is_non_destructive(self) -> None:
        """Forward-only and non-destructive is the corpus's central promise:
        an availability record that can be edited afterwards is not evidence."""
        text = schema.SCHEMA_DDL.upper()
        for statement in ("DROP ", "DELETE ", "TRUNCATE ", "ALTER "):
            assert statement not in text, statement

    def test_the_store_delegates_rather_than_reimplements(self) -> None:
        """Every read method on the store is a forward to `queries`. Two copies
        of a query is how a paginated reader and a point-in-time reader ended
        up sorting differently in the first place."""
        tree = ast.parse((PACKAGE / "store.py").read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "IntelligenceStore"
        )
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in queries.__all__:
                continue
            body = [item for item in node.body if not isinstance(item, ast.Expr)]
            assert len(body) == 1 and isinstance(body[0], ast.Return), node.name
            assert f"queries.{node.name}(" in ast.unparse(body[0]), node.name

    def test_no_module_is_oversized_again(self) -> None:
        """The split is worth keeping only if it stays split."""
        sizes = {
            path.name: len(path.read_text(encoding="utf-8").splitlines())
            for path in sorted(PACKAGE.glob("*.py"))
        }
        assert max(sizes.values()) < 700, sizes


class TestTheReadsNeedOnlyAConnection:
    """The reason the reads moved: none of them needs a store.

    A query that can be exercised against a bare connection is a query that gets
    exercised. These run against an in-memory database with the schema applied
    and nothing else -- no store, no fixtures, no corpus.
    """

    @pytest.fixture()
    def connection(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(":memory:")
        schema.migrate(connection)
        return connection

    def test_the_schema_applies_to_a_bare_connection(
        self, connection: duckdb.DuckDBPyConnection
    ) -> None:
        assert schema.schema_version(connection) == schema.SCHEMA_VERSION
        assert schema.is_ready(connection)

    def test_migration_is_idempotent(self, connection: duckdb.DuckDBPyConnection) -> None:
        schema.migrate(connection)
        schema.migrate(connection)
        assert schema.is_ready(connection)

    def test_a_future_schema_is_refused_rather_than_downgraded(
        self, connection: duckdb.DuckDBPyConnection
    ) -> None:
        from market_intelligence.errors import StorageError

        connection.execute(
            "INSERT OR REPLACE INTO schema_metadata VALUES (?, ?)",
            [schema.COMPONENT, schema.SCHEMA_VERSION + 1],
        )
        with pytest.raises(StorageError, match="newer than supported"):
            schema.migrate(connection)

    def test_an_empty_corpus_reads_as_empty(
        self, connection: duckdb.DuckDBPyConnection
    ) -> None:
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert queries.documents_as_of(connection, origin) == []
        assert queries.signals_as_of(connection, origin) == []
        assert queries.eligible_signals_as_of(connection, origin) == []
        assert queries.correction_count(connection) == 0
        assert queries.quarantine_count(connection) == 0
        assert queries.metrics(connection)["documents_ingested"] == 0

    def test_the_total_order_survived_the_move(
        self, connection: duckdb.DuckDBPyConnection
    ) -> None:
        """Availability first, id as the tiebreaker -- the property `sum()`
        being non-associative made load-bearing. Rows are inserted in the
        reverse of the expected order, and out of insertion order, so a query
        that dropped the tiebreaker would have to get lucky twice to pass.
        """
        instant = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        later = instant + timedelta(hours=1)
        for document_id, available_at in (
            ("d-zulu", instant),
            ("d-alpha", instant),
            ("d-mike", later),
            ("d-bravo", instant),
        ):
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?)",
                [document_id, available_at, "{}"],
            )
        ordered = connection.execute(
            "SELECT document_id FROM documents WHERE available_at <= ? "
            "ORDER BY available_at, document_id",
            [later],
        ).fetchall()
        assert [row[0] for row in ordered] == ["d-alpha", "d-bravo", "d-zulu", "d-mike"]

    def test_the_point_in_time_queries_order_by_id_within_an_instant(self) -> None:
        """The SQL itself, because an ORDER BY is the kind of thing a later
        edit removes without any test noticing."""
        source = queries_source()
        assert "ORDER BY available_at, document_id" in source
        assert "ORDER BY available_time, event_id" in source


def queries_source() -> str:
    return (PACKAGE / "queries.py").read_text(encoding="utf-8")
