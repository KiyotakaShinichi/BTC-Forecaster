"""B5.2 — a deployment's configuration is checked, and its feeds probed, before it collects.

The collector already refuses a profile that asks for a contact it was not
given. What an operator installing it on a host did not have was a way to ask,
before the first 03:07 cycle, whether the host is deployable at all -- and
whether each feed can actually be read by *this* collector, with *this*
User-Agent, from *this* network.

Pinned here:

* `ops-config-check` is deployable exactly when the profile loads with a
  contact, enables no retired feed and has a usable state root, and never prints
  the contact address;
* `ops-probe` reads each configured feed once, with no retries, through the
  collector's own opener; it sends the configured contact, reports each feed's
  outcome and failure class, prints no contact, and writes nothing;
* secrets never reach a log line, by name or by value;
* collection and the probe build the provider one way.

No network.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from market_intelligence.cli import main
from market_intelligence.collection import syndication
from market_intelligence.collection.backoff import FailureClass, ProviderFailure
from market_intelligence.logs import configure, get_logger, redact
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import SYNDICATION, CollectionProfile
from tests.test_intelligence_deployment import DEPLOYED_PROFILE, FEED_PAYLOAD

CONTACT = "collector-ops@example.org"
UNPARSEABLE = b"<rss><channel><item><title>x</channel></rss>"


def profile_file(tmp_path: Path, **overrides: object) -> Path:
    raw: dict[str, object] = {
        "name": "b52-test",
        "user_agent": "BTC-Forecaster Research <${BTC_INTEL_CONTACT}>",
        "feeds": ["sec-press", "cftc-press"],
        "watchlist": [
            {
                "canonical_name": "SEC",
                "aliases": ["Securities and Exchange Commission"],
                "entity_type": "REGULATOR",
                "topics": ["bitcoin regulation"],
                "expected_event_types": ["REGULATION"],
            }
        ],
    }
    raw.update(overrides)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def contact(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("BTC_INTEL_CONTACT", CONTACT)
    return CONTACT


@pytest.fixture
def state(tmp_path: Path) -> Path:
    return StoragePaths.from_environment(tmp_path / "state").ensure().root


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestConfigCheck:
    def test_the_deployed_profile_is_deployable_once_a_contact_is_set(
        self, contact: str, state: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(
            ["ops-config-check", "--profile", str(DEPLOYED_PROFILE), "--state-root", str(state), "--json"], capsys
        )
        payload = json.loads(out)
        assert code == 0 and payload["deployable"], payload["problems"]
        assert payload["profile"]["user_agent"] == "BTC-Forecaster Research <<redacted>>"
        assert not any(feed["retired"] for feed in payload["profile"]["feeds"])
        assert payload["profile"]["minimum_interval_seconds"] >= payload["profile"]["declared_floor_seconds"]
        assert payload["profile"]["retry"]["retried_failure_classes"] == ["RATE_LIMIT", "TRANSIENT"]
        assert CONTACT not in out + err

    def test_the_human_view_carries_no_contact_either(
        self, contact: str, state: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(["ops-config-check", "--profile", str(DEPLOYED_PROFILE), "--state-root", str(state)], capsys)
        assert code == 0 and "deployable   yes" in out
        assert CONTACT not in out + err

    def test_without_a_contact_it_is_not_deployable(
        self, monkeypatch: pytest.MonkeyPatch, state: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("BTC_INTEL_CONTACT", raising=False)
        code, out, err = run(
            ["ops-config-check", "--profile", str(DEPLOYED_PROFILE), "--state-root", str(state), "--json"], capsys
        )
        payload = json.loads(out)
        assert code == 2 and not payload["deployable"]
        assert any("BTC_INTEL_CONTACT" in problem for problem in payload["problems"])
        assert "Traceback" not in out + err

    def test_a_retired_feed_is_not_deployable(
        self, contact: str, state: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = profile_file(tmp_path, feeds=["sec-press", "sec-litigation"])
        code, out, _ = run(["ops-config-check", "--profile", str(path), "--state-root", str(state), "--json"], capsys)
        assert code == 2
        assert any("sec-litigation" in problem for problem in json.loads(out)["problems"])

    def test_a_profile_with_no_contact_user_agent_is_not_deployable(
        self, contact: str, state: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = profile_file(tmp_path, user_agent=None)
        code, out, _ = run(["ops-config-check", "--profile", str(path), "--state-root", str(state), "--json"], capsys)
        assert code == 2
        assert any("no contact User-Agent" in problem for problem in json.loads(out)["problems"])

    def test_a_contact_written_into_the_profile_is_not_echoed(self, tmp_path: Path) -> None:
        profile = CollectionProfile.load(profile_file(tmp_path, user_agent="BTC-Forecaster Research <literal@example.org>"))
        assert profile.describe()["user_agent"] == "<set in the profile itself; not shown>"


def serving(responses: dict[str, object]):
    """An opener answering by host: bytes are served, an exception is raised. Records every call."""
    calls: list[str] = []

    def fetch(url: str, timeout: float, agent: str | None = None) -> bytes:
        calls.append(url)
        host = url.split("/")[2]
        answer = next(value for key, value in responses.items() if key in host)
        if isinstance(answer, BaseException):
            raise answer
        assert isinstance(answer, bytes)
        return answer

    return fetch, calls


class TestProbe:
    def probe(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict, str]:
        code, out, err = run(["ops-probe", "--profile", str(profile_file(tmp_path)), "--json"], capsys)
        return code, json.loads(out) if out.strip() else {}, out + err

    def test_every_feed_readable(
        self, contact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetch, calls = serving({"sec.gov": FEED_PAYLOAD, "cftc.gov": FEED_PAYLOAD})
        monkeypatch.setattr(syndication, "_default_opener", fetch)
        code, payload, text = self.probe(tmp_path, capsys)
        assert code == 0 and (payload["readable"], payload["configured"]) == (2, 2)
        assert [row["entries"] for row in payload["feeds"]] == [1, 1]
        assert payload["user_agent"] == "BTC-Forecaster Research <<redacted>>"
        assert len(calls) == 2 and CONTACT not in text

    def test_a_failing_feed_is_named_with_its_class(
        self, contact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetch, _ = serving(
            {"sec.gov": FEED_PAYLOAD, "cftc.gov": ProviderFailure(FailureClass.PERMANENT, "HTTP 404 for the CFTC feed")}
        )
        monkeypatch.setattr(syndication, "_default_opener", fetch)
        code, payload, _ = self.probe(tmp_path, capsys)
        cftc = next(row for row in payload["feeds"] if row["feed_id"] == "cftc-press")
        assert code == 1 and payload["readable"] == 1
        assert (cftc["ok"], cftc["failure_class"]) == (False, "PERMANENT")

    def test_no_readable_feed_is_exit_2_and_nothing_is_retried(
        self, contact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outage = ProviderFailure(FailureClass.TRANSIENT, "connection reset")
        fetch, calls = serving({"sec.gov": outage, "cftc.gov": outage})
        monkeypatch.setattr(syndication, "_default_opener", fetch)
        code, payload, _ = self.probe(tmp_path, capsys)
        assert code == 2 and payload["readable"] == 0
        assert len(calls) == 2, "a probe retried a transient failure"

    def test_an_unparseable_feed_is_a_schema_failure(
        self, contact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetch, _ = serving({"sec.gov": UNPARSEABLE, "cftc.gov": UNPARSEABLE})
        monkeypatch.setattr(syndication, "_default_opener", fetch)
        code, payload, _ = self.probe(tmp_path, capsys)
        assert code == 2
        assert {row["failure_class"] for row in payload["feeds"]} == {"SCHEMA"}

    def test_the_probe_sends_the_configured_contact_and_prints_none(
        self, contact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sent: list[str | None] = []

        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def read(self) -> bytes:
                return FEED_PAYLOAD

        def urlopen(request: object, timeout: float) -> Response:
            sent.append(request.get_header("User-agent"))  # type: ignore[attr-defined]
            return Response()

        monkeypatch.setattr(syndication.urllib.request, "urlopen", urlopen)
        code, _, text = self.probe(tmp_path, capsys)
        assert code == 0
        assert sent == [f"BTC-Forecaster Research <{CONTACT}>"] * 2
        assert CONTACT not in text

    def test_the_probe_writes_nothing(
        self, contact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fetch, _ = serving({"sec.gov": FEED_PAYLOAD, "cftc.gov": FEED_PAYLOAD})
        monkeypatch.setattr(syndication, "_default_opener", fetch)
        monkeypatch.setenv("BTC_INTEL_STATE_ROOT", str(tmp_path / "never-state"))
        database = tmp_path / "never.duckdb"
        code, _, _ = run(["--db", str(database), "ops-probe", "--profile", str(profile_file(tmp_path))], capsys)
        assert code == 0
        assert not database.exists() and not (tmp_path / "never-state").exists()


class TestSecretsStayOnTheHost:
    def test_a_secret_is_dropped_by_name_and_scrubbed_by_value(self) -> None:
        cleaned = redact(
            {
                "detail": f"HTTP 403 for a request from {CONTACT}",
                "reasons": [f"blocked: {CONTACT}"],
                "authorization": "Bearer abc123",
                "api_key": "k-123",
                "user_agent": f"BTC-Forecaster Research <{CONTACT}>",
                "token": "t-1",
                "run": "ok",
            },
            environment={"BTC_INTEL_CONTACT": CONTACT},
        )
        rendered = json.dumps(cleaned)
        assert set(cleaned) == {"detail", "reasons", "run"}
        assert CONTACT not in rendered and "abc123" not in rendered and "k-123" not in rendered

    def test_the_collectors_logger_writes_no_secret(self, contact: str) -> None:
        stream = io.StringIO()
        configure(stream=stream, level="INFO")
        get_logger("b52").emit(
            "provider_result", detail=f"403 while identifying as {CONTACT}", authorization="Bearer abc123"
        )
        text = stream.getvalue()
        assert "provider_result" in text, "the record was not written, so this test would prove nothing"
        assert CONTACT not in text and "abc123" not in text


class TestOneProviderConstruction:
    def test_the_probe_and_the_cycle_build_the_same_provider(self, contact: str) -> None:
        profile = CollectionProfile.load(DEPLOYED_PROFILE)
        probed = profile.syndication_provider()
        collected = profile.build_retriever([SYNDICATION]).providers[SYNDICATION]
        for provider in (probed, collected):
            assert provider.user_agent == f"BTC-Forecaster Research <{CONTACT}>"  # type: ignore[attr-defined]
            assert provider.timeout == profile.timeout  # type: ignore[attr-defined]
            assert [feed.feed_id for feed in provider.feeds] == [feed.feed_id for feed in profile.feeds]  # type: ignore[attr-defined]
