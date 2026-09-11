"""The CLI's contract, held across the split into `commands/`.

`_main` used to be a 230-line chain of `if args.command == ...`. The commands
now live in six modules behind a registry, and what has to survive that is not
"it still works" but three specific promises an operator and a scheduler both
depend on:

* the command set is unchanged -- every name that ran before still runs;
* the exit codes are unchanged -- 0 ran, 2 failed, 3 locked, 4 nothing due,
  and a scheduler branches on those without parsing anything;
* the parser and the registry agree -- a command declared to argparse and never
  registered would parse cleanly and then quietly return 1.

The last one is what a registry buys over a chain: it is checkable.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

from market_intelligence import cli
from market_intelligence.commands import _registry, command_names, dispatch

#: Captured before the split. Written out rather than derived from the parser,
#: because a list derived from the parser would agree with whatever the parser
#: happens to declare.
COMMANDS_BEFORE_THE_SPLIT = frozenset({
    "aggregate", "backfill", "catalog", "collect",
    "collect-scheduled", "corpora", "corpus-backup", "corpus-correct",
    "corpus-corrections", "corpus-restore", "corpus-status", "corpus-verify",
    "dataset", "demo", "extract", "gold-report",
    "health", "ops-paths", "ops-status", "ops-watch",
    "providers", "quality", "replay", "replay-dataset",
})

#: Added deliberately by B5.2 -- operations commands for a production host. Each
#: manages its own storage or reads none, so none of them is handed a store.
COMMANDS_ADDED_IN_B52 = frozenset({"ops-config-check", "ops-probe", "ops-backup", "corpus-backup-verify"})


def parser_commands() -> frozenset[str]:
    parser = cli.build_parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    return frozenset(action.choices)


class TestEveryCommandSurvived:
    def test_the_parser_still_declares_all_of_them(self) -> None:
        missing = COMMANDS_BEFORE_THE_SPLIT - parser_commands()
        assert not missing, f"lost from the parser: {sorted(missing)}"

    def test_the_registry_covers_exactly_the_parser(self) -> None:
        """Both directions. A command in the parser with no handler returns 1
        for reasons no error message explains; a handler with no parser entry
        is unreachable code that looks live."""
        assert parser_commands() == frozenset(command_names())

    def test_no_command_was_added_unannounced(self) -> None:
        added = parser_commands() - COMMANDS_BEFORE_THE_SPLIT - COMMANDS_ADDED_IN_B52
        assert not added, f"new commands, update the frozen list deliberately: {sorted(added)}"

    def test_the_help_still_lists_them_in_one_place(self, capsys: pytest.CaptureFixture) -> None:
        """Parser composition stayed in `cli.py` for this: `--help` assembled
        from fragments each command contributes is help nobody can read whole."""
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--help"])
        rendered = capsys.readouterr().out
        for name in ("collect-scheduled", "corpus-verify", "ops-watch"):
            assert name in rendered, name


class TestTheStoreLifecycleIsUnchanged:
    def test_the_commands_that_never_had_a_store_still_do_not_get_one(self) -> None:
        """Opening a store creates the database file. `demo` and `gold-report`
        run on fixtures, and one that left a stray empty corpus in the current
        directory would be a demo with a side effect."""
        storeless, with_store = _registry()
        assert set(storeless) == {"demo", "gold-report", "collect-scheduled"} | COMMANDS_ADDED_IN_B52
        assert not set(storeless) & set(with_store)

    def test_a_storeless_command_creates_no_database(self, tmp_path: Path) -> None:
        args = cli.build_parser().parse_args(
            ["--db", str(tmp_path / "should-not-exist.duckdb"), "gold-report",
             "--output", str(tmp_path / "gold.json")]
        )
        assert dispatch(args) == 0
        assert not (tmp_path / "should-not-exist.duckdb").exists()

    def test_a_store_command_closes_what_it_opened(self, tmp_path: Path) -> None:
        database = tmp_path / "corpus.duckdb"
        assert cli.main(["--db", str(database), "health"]) == 0
        assert database.exists()
        # Closed, so a second invocation can open the same file. DuckDB holds an
        # exclusive lock; a leaked handle makes the next run fail on a corpus
        # that is perfectly fine.
        assert cli.main(["--db", str(database), "health"]) == 0


class TestExitCodesAreUnchanged:
    """0 ran, 2 failed, 3 another collector holds the lock, 4 nothing was due.

    Checked against real invocations rather than by reading the handlers,
    because the value a scheduler sees is the one `main` returns.
    """

    def test_an_ordinary_command_returns_zero(self, tmp_path: Path) -> None:
        assert cli.main(["--db", str(tmp_path / "c.duckdb"), "providers"]) == 0

    def test_verify_fails_closed_on_an_empty_corpus(self, tmp_path: Path) -> None:
        """An empty corpus is verifiable and therefore fine; what matters is
        that the code is the report, not a sentence beside it."""
        code = cli.main(["--db", str(tmp_path / "c.duckdb"), "corpus-verify", "--json"])
        assert code in (0, 2)

    def test_a_missing_correction_file_exits_two_with_a_sentence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = cli.main([
            "--db", str(tmp_path / "c.duckdb"),
            "corpus-correct", "--file", str(tmp_path / "absent.json"),
        ])
        assert code == 2
        error = capsys.readouterr().err
        assert error.startswith("error: ")
        assert "Traceback" not in error

    def test_a_malformed_origins_file_exits_two_rather_than_raising(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """This one changed for the better. A naive timestamp in the origins
        file used to escape as an argparse exception and exit 1 -- a code the
        documented contract does not mention."""
        origins = tmp_path / "origins.txt"
        origins.write_text("2026-01-01T00:00:00\n", encoding="utf-8")
        code = cli.main([
            "--db", str(tmp_path / "c.duckdb"), "dataset",
            "--origins", str(origins),
            "--output", str(tmp_path / "out.parquet"),
            "--manifest", str(tmp_path / "out.json"),
            "--config-fingerprint", "test",
        ])
        assert code == 2
        assert "must include a timezone" in capsys.readouterr().err

    def test_an_unknown_command_is_refused_by_the_parser(self) -> None:
        with pytest.raises(SystemExit) as raised:
            cli.main(["not-a-command"])
        assert raised.value.code == 2


class TestTheModuleBoundariesHold:
    def test_the_dispatch_chain_is_gone(self) -> None:
        """The thing this split existed to remove. A reintroduced chain would
        work and would quietly put the registry out of date."""
        tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
        comparisons = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "command"
        ]
        assert not comparisons, f"{len(comparisons)} branches on args.command remain"

    def test_cli_holds_only_the_parser_and_the_entry_point(self) -> None:
        tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
        defined = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert defined == {"_time", "build_parser", "main"}, sorted(defined)

    def test_the_report_builders_still_resolve_by_their_old_names(self) -> None:
        """`api.py`, the market-intelligence workflow and four test modules
        import these from `cli`. They now live in `reports`, and the old names
        are kept because renaming a symbol in use is a break, not a tidy-up."""
        from market_intelligence import reports

        assert cli._corpus_status is reports.corpus_status
        assert cli._provider_report is reports.provider_report
        assert cli._ops_report is reports.ops_report

    def test_the_reports_no_longer_depend_on_the_parser(self) -> None:
        """The reason they moved. Answering "what does the corpus hold" should
        not require importing argparse."""
        source = Path(cli.__file__).parent.joinpath("reports.py").read_text(encoding="utf-8")
        assert "import argparse" not in source
        assert "from .cli" not in source

    def test_no_handler_exits_the_process(self) -> None:
        """Exit codes are returned so `main` owns the contract. A handler that
        calls `sys.exit` bypasses it and cannot be tested in-process."""
        package = Path(cli.__file__).parent / "commands"
        for path in sorted(package.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "sys.exit" not in source, path.name
            assert "SystemExit" not in source, path.name

    def test_no_command_module_is_oversized(self) -> None:
        package = Path(cli.__file__).parent / "commands"
        sizes = {
            path.name: len(path.read_text(encoding="utf-8").splitlines())
            for path in sorted(package.glob("*.py"))
        }
        assert max(sizes.values()) < 250, sizes
        assert len(Path(cli.__file__).read_text(encoding="utf-8").splitlines()) < 300


class TestTheCommandsStillDoTheirWork:
    """A handful end to end, because a registry that dispatches correctly to a
    broken handler is still broken."""

    def test_gold_report_writes_its_evaluation(self, tmp_path: Path) -> None:
        output = tmp_path / "gold.json"
        assert cli.main(["gold-report", "--output", str(output)]) == 0
        assert json.loads(output.read_text(encoding="utf-8"))

    def test_corpus_status_reports_not_ready_on_an_empty_corpus(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = cli.main([
            "--db", str(tmp_path / "c.duckdb"), "corpus-status", "--json",
            "--entities", "SEC",
        ])
        assert code == 0
        assert json.loads(capsys.readouterr().out)["readiness"] == "NOT_READY"

    def test_ops_paths_reports_where_state_would_live(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = cli.main([
            "--db", str(tmp_path / "c.duckdb"), "ops-paths",
            "--state-root", str(tmp_path / "state"),
        ])
        payload = json.loads(capsys.readouterr().out)
        assert code in (0, 2)
        assert "database" in payload["paths"]

    def test_corrections_report_is_empty_and_honest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = cli.main([
            "--db", str(tmp_path / "c.duckdb"), "corpus-corrections", "--json",
        ])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["corrections_recorded"] == 0
        assert payload["events_excluded"] == 0
