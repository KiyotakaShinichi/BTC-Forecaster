"""Structured logs that reach somebody, and carry no secret when they do.

There was already a structured logger. It emitted JSON, it dropped fields whose
names looked like credentials, and two of the three surfaces that logged
anything used it. What it did not have was a handler -- nothing in the
repository ever attached one -- so every INFO record fell through to the root
logger's last-resort handler at WARNING and was discarded. Structured logs
nobody can read are documentation, not observability.

These tests pin the three things that fixes:

* a configured destination, idempotent, on stderr rather than stdout;
* the same emitter on every surface, with the same field names, so grouping by
  `run_id` works across the collector, the API and the replay service;
* redaction by *value* as well as by name -- the check that matters, because a
  contact address reaches a log through `detail` far more easily than through a
  field somebody named `contact`.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from market_intelligence import logs
from market_intelligence.cycle import StructuredRunLogger


@pytest.fixture()
def captured() -> tuple[io.StringIO, logs.StructuredLogger]:
    stream = io.StringIO()
    logger = logging.getLogger("btc_intelligence.test_capture")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return stream, logs.StructuredLogger(logger)


def records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


class TestEveryRecordIsOneJsonObject:
    def test_a_record_parses(self, captured) -> None:
        stream, log = captured
        log.emit("run_started", run_id="abc", queries=3)
        (record,) = records(stream)
        assert record["event"] == "run_started"
        assert record["run_id"] == "abc"
        assert record["queries"] == 3

    def test_every_record_carries_a_timestamp_and_a_severity(self, captured) -> None:
        """Without them a JSON line is a dict, not a log entry."""
        stream, log = captured
        log.emit("thing")
        log.emit("problem", severity=logging.WARNING)
        first, second = records(stream)
        assert first["severity"] == "INFO" and second["severity"] == "WARNING"
        assert first["ts"].endswith("+00:00"), first["ts"]

    def test_the_contract_fields_keep_their_names(self, captured) -> None:
        """`run_id`, `provider_id` and `origin` are how records from three
        different surfaces get grouped. They are named parameters rather than
        loose kwargs so a caller who writes `runid` gets an ordinary field and
        not a silently broken correlation key."""
        from datetime import datetime, timezone

        stream, log = captured
        log.emit(
            "provider_result",
            run_id="r1",
            provider_id="syndication",
            origin=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        (record,) = records(stream)
        assert record["run_id"] == "r1"
        assert record["provider_id"] == "syndication"
        assert record["origin"] == "2026-03-01T00:00:00+00:00"

    def test_severity_is_the_log_level_too(self, captured) -> None:
        """So `journalctl -p warning` and a `jq` filter agree."""
        stream, log = captured
        log.logger.setLevel(logging.WARNING)
        log.emit("quiet", severity=logging.INFO)
        log.emit("loud", severity=logging.ERROR)
        assert [record["event"] for record in records(stream)] == ["loud"]


class TestNothingSecretReachesALog:
    SECRET = "collector@example.invalid"
    ENVIRONMENT = {"BTC_INTEL_CONTACT": SECRET, "BTC_INTEL_SEARCH_API_KEY": "sk-live-9f2a"}

    def test_fields_named_like_a_secret_are_dropped(self) -> None:
        cleaned = logs.redact(
            {"provider": "p", "api_key": "sk-live", "authorization": "Bearer x", "credential": "c"},
            environment={},
        )
        assert cleaned == {"provider": "p"}

    def test_the_user_agent_is_dropped_because_it_carries_the_contact(self) -> None:
        """The collector's User-Agent is `BTC-Forecaster Research <address>`.
        Logging it would publish the one value the deployment was explicitly
        told never to commit."""
        cleaned = logs.redact({"user_agent": "BTC-Forecaster Research <a@b.c>"}, environment={})
        assert cleaned == {}

    def test_a_secret_value_is_scrubbed_from_a_field_that_is_not_named_like_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure name-based filtering cannot catch. An address travels
        into a log inside `detail`, `reason` or an exception message far more
        readily than inside a field somebody called `contact`."""
        cleaned = logs.redact(
            {"detail": f"403 from sec.gov for {self.SECRET}"}, environment=self.ENVIRONMENT
        )
        assert self.SECRET not in cleaned["detail"]
        assert logs.REDACTED in cleaned["detail"]

    def test_scrubbing_reaches_into_nested_structures(self) -> None:
        cleaned = logs.redact(
            {"providers": [{"note": f"contact {self.SECRET}"}]}, environment=self.ENVIRONMENT
        )
        assert self.SECRET not in json.dumps(cleaned)

    def test_a_short_value_is_not_treated_as_a_secret(self) -> None:
        """A one-character contact would otherwise scrub a letter out of every
        line in the journal."""
        cleaned = logs.redact({"detail": "abc"}, environment={"BTC_INTEL_CONTACT": "a"})
        assert cleaned["detail"] == "abc"

    def test_the_emitted_record_is_redacted_end_to_end(self, captured) -> None:
        stream, log = captured
        log.emit("attempt", provider_id="p", api_key="sk-live-secret", authorization="Bearer t")
        (record,) = records(stream)
        assert record["provider_id"] == "p"
        assert "sk-live-secret" not in json.dumps(record)
        assert "api_key" not in record

    def test_every_declared_secret_variable_is_covered(self) -> None:
        """`.env.example` documents five secret-ish variables for the collector;
        all five have to be scrubbed, not just the one that broke first. The
        fifth, B5.2's alert command, can carry a webhook token in its text."""
        assert set(logs.SECRET_ENV_VARIABLES) == {
            "BTC_INTEL_ALERT_COMMAND",
            "BTC_INTEL_CONTACT",
            "BTC_INTEL_SEARCH_API_KEY",
            "BTC_INTEL_STATEMENTS_API_KEY",
            "BTC_INTEL_WHALE_API_KEY",
        }


class TestTheDestinationIsConfigured:
    def test_configure_attaches_exactly_one_handler(self) -> None:
        stream = io.StringIO()
        logs.configure(stream=stream, level=logging.INFO)
        logs.configure(stream=stream, level=logging.INFO)
        logger = logging.getLogger(logs.LOGGER_NAME)
        structured = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_btc_intel_structured", False)
        ]
        assert len(structured) == 1
        logs.get_logger("test").emit("once")
        assert len(records(stream)) == 1

    def test_records_do_not_propagate_to_the_root_logger(self) -> None:
        """An embedding application's root handler would print every line a
        second time, in its own format."""
        logs.configure(stream=io.StringIO())
        assert logging.getLogger(logs.LOGGER_NAME).propagate is False

    def test_the_level_comes_from_the_environment(self) -> None:
        stream = io.StringIO()
        logs.configure(stream=stream, environment={logs.LEVEL_ENV: "WARNING"})
        logs.get_logger("test").emit("quiet")
        logs.get_logger("test").emit("loud", severity=logging.ERROR)
        assert [record["event"] for record in records(stream)] == ["loud"]

    def test_the_default_is_info(self) -> None:
        logs.configure(stream=io.StringIO(), environment={})
        assert logging.getLogger(logs.LOGGER_NAME).level == logging.INFO

    def test_nothing_configures_logging_on_import(self) -> None:
        """A library that attaches a handler when imported takes the
        destination away from whatever imported it. Configuration belongs to
        the two entry points that are always non-interactive: the scheduled
        collector and the API server."""
        import ast

        package = Path(logs.__file__).parent
        offenders = []
        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "basicConfig":
                    offenders.append(f"{path.name}: basicConfig")
            # A configure() at module scope runs on import, which is the same
            # defect wearing this repository's own function name.
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if getattr(node.value.func, "id", "") == "configure":
                        offenders.append(f"{path.name}: configure at import")
        assert not offenders, offenders

    def test_it_is_configured_where_it_matters(self) -> None:
        scheduled = (
            Path(logs.__file__).parent / "commands" / "collection.py"
        ).read_text(encoding="utf-8")
        assert "configure()" in scheduled
        server = (Path(logs.__file__).parents[1] / "btc-intel-api.py").read_text(encoding="utf-8")
        assert "configure()" in server

    def test_diagnostics_go_to_stderr_not_stdout(self) -> None:
        """`btc-intel <command>` writes its result to stdout for a caller to
        pipe into `jq`. Diagnostics interleaved there would corrupt it."""
        source = Path(logs.__file__).read_text(encoding="utf-8")
        assert "sys.stderr" in source
        assert "sys.stdout" not in source


class TestTheOperationalPathsEmit:
    def test_the_scheduler_path_reports_what_it_decided(self) -> None:
        """It emitted nothing at all before. "Ran and collected zero" and "did
        not run" are the two outcomes an operator most needs to tell apart, and
        an exit code alone does not say which providers were involved."""
        source = (
            Path(logs.__file__).parent / "ops" / "scheduled.py"
        ).read_text(encoding="utf-8")
        for event in (
            "collection_started",
            "collection_skipped",
            "collection_not_due",
            "collection_finished",
            "provider_result",
            "storage_ephemeral",
        ):
            assert f'"{event}"' in source, event

    def test_a_failed_provider_is_a_warning_not_information(self) -> None:
        """The single most common cause of a green run with an empty corpus."""
        source = (
            Path(logs.__file__).parent / "ops" / "scheduled.py"
        ).read_text(encoding="utf-8")
        assert "logging.INFO if succeeded else logging.WARNING" in source

    def test_the_old_emitter_name_still_works(self) -> None:
        """`cycle.StructuredRunLogger` is imported by name in the existing
        operations tests. It is the same class now."""
        assert StructuredRunLogger is logs.StructuredLogger

    def test_no_surface_still_builds_a_logfmt_string(self) -> None:
        """`services.py` and `api.py` emitted `key=value` text through
        `logging` directly, so half the output was JSON and half was not."""
        package = Path(logs.__file__).parent
        for name in ("services.py", "api.py", "cycle.py"):
            source = (package / name).read_text(encoding="utf-8")
            assert "logging.getLogger" not in source, name
