"""Deployment — the committed profile, and the one command a scheduler calls.

The profile is the seam between "the code is correct" and "the deployment is
correct", and it is where a working collector quietly becomes a useless one: a
feed dropped from a list, a watchlist edited on the host and never committed, a
licensed provider that looks enabled because its variable name is spelled right
in a config file while nothing ever set the variable.

So the tests that matter here are the ones that fail when configuration lies
about itself. The committed deployment profile is loaded for real, not mocked --
if `deploy/collection-profile.json` stops being deployable, this file goes red
before a host does.

No network: the one end-to-end cycle serves a feed from memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_intelligence.collection import syndication
from market_intelligence.collection.feeds import NEWS_API_DECLARATION, SYNDICATION_DECLARATION
from market_intelligence.errors import ConfigurationError
from market_intelligence.ops.paths import StoragePaths
from market_intelligence.ops.profile import SYNDICATION, CollectionProfile, collect_once
from market_intelligence.ops.runlock import RunLock
from market_intelligence.ops.scheduled import EXIT_LOCK_HELD, EXIT_NOTHING_DUE, EXIT_OK
from market_intelligence.storage import IntelligenceStore

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)

#: The profile that is actually deployed, not a fixture resembling it.
DEPLOYED_PROFILE = Path(__file__).resolve().parents[1] / "deploy" / "collection-profile.json"

FEED_PAYLOAD = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Commission Press Releases</title>
  <item>
    <title>Securities and Exchange Commission announces bitcoin regulation decision</title>
    <link>https://sec.example.gov/news/2026/decision</link>
    <description>The Securities and Exchange Commission announced a decision.</description>
    <pubDate>Wed, 02 Sep 2026 09:30:00 GMT</pubDate>
  </item>
</channel></rss>"""


def minimal(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "name": "test-profile",
        "feeds": ["sec-press"],
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
    return raw


# ------------------------------------------------------------ the real profile


class TestTheCommittedProfile:
    """`deploy/collection-profile.json` is deployable, or this is red."""

    #: The deployed profile asks the host for a contact address, so loading it
    #: needs one. A literal stands in here: the real address is the operator's
    #: and belongs in the host environment, never in the repository.
    ENVIRONMENT = {"BTC_INTEL_CONTACT": "tests@example.org"}

    def deployed(self) -> CollectionProfile:
        return CollectionProfile.load(DEPLOYED_PROFILE, environment=self.ENVIRONMENT)

    def test_the_deployed_profile_demands_a_contact_address(self) -> None:
        """A collector polling government feeds for months without one gets
        blocked, weeks in, as a PERMANENT failure nobody expected."""
        with pytest.raises(ConfigurationError, match="BTC_INTEL_CONTACT"):
            CollectionProfile.load(DEPLOYED_PROFILE, environment={})

    def test_the_deployed_profile_carries_no_address_of_its_own(self) -> None:
        """The shape is committed; the address is not. A repository is the wrong
        place to publish somebody's email."""
        raw = DEPLOYED_PROFILE.read_text(encoding="utf-8")
        assert "${BTC_INTEL_CONTACT}" in raw
        assert "@" not in json.loads(raw)["user_agent"].replace("${BTC_INTEL_CONTACT}", "")

    def test_the_contact_address_reaches_the_request_but_not_the_manifest(self) -> None:
        profile = self.deployed()
        assert profile.user_agent == "BTC-Forecaster Research <tests@example.org>"
        assert "tests@example.org" not in json.dumps(profile.fingerprint())

    def test_the_deployed_profile_loads(self) -> None:
        profile = self.deployed()
        assert profile.name
        assert profile.feeds, "a profile with no feeds collects nothing"
        assert profile.entities

    def test_every_deployed_feed_is_in_the_committed_catalogue(self) -> None:
        """A feed id is resolved against `feeds.py`, so this cannot drift."""
        profile = self.deployed()
        assert {feed.feed_id for feed in profile.feeds} == set(profile.fingerprint()["feeds"])

    def test_the_deployed_profile_watches_official_primary_sources(self) -> None:
        """B4's hypotheses are about what the regulator did, not who reported it."""
        profile = self.deployed()
        assert all(feed.official_source for feed in profile.feeds)
        assert sum(feed.primary_source for feed in profile.feeds) >= 1

    def test_the_deployed_profile_respects_the_declared_poll_floor(self) -> None:
        profile = self.deployed()
        assert profile.minimum_interval_seconds >= SYNDICATION_DECLARATION.minimum_interval_seconds

    def test_the_deployed_profile_spans_several_publishers(self) -> None:
        """O21: the readiness gate counts distinct publishers, so one feed family
        collecting forever can never open it however long it runs."""
        profile = self.deployed()
        assert len({feed.publisher for feed in profile.feeds}) >= 3


# --------------------------------------------------------- refusing bad config


class TestConfigurationIsRefusedNotGuessed:
    def test_a_feed_outside_the_catalogue_is_refused(self) -> None:
        """The allowlist is the policy. An operator cannot widen reach by
        editing JSON on the host -- that is a reviewed change to feeds.py."""
        with pytest.raises(ConfigurationError, match="committed"):
            CollectionProfile.from_mapping(minimal(feeds=["sec-press", "an-arbitrary-blog"]))

    def test_polling_faster_than_the_declaration_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="faster than"):
            CollectionProfile.from_mapping(minimal(minimum_interval_seconds=30))

    def test_a_user_agent_with_no_contact_is_refused(self) -> None:
        """A publisher that decides to block this collector needs somewhere to
        write first; a bare product string gives them nowhere."""
        with pytest.raises(ConfigurationError, match="contact address"):
            CollectionProfile.from_mapping(minimal(user_agent="btc-intel/2.0"))

    @pytest.mark.parametrize(
        "agent",
        ["btc-intel/2.0 (research@example.org)", "btc-intel/2.0 (https://example.org/contact)"],
    )
    def test_a_contactable_user_agent_is_accepted(self, agent: str) -> None:
        assert CollectionProfile.from_mapping(minimal(user_agent=agent)).user_agent == agent

    def test_an_empty_feed_list_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="no feeds"):
            CollectionProfile.from_mapping(minimal(feeds=[]))

    def test_a_watchlist_with_everything_disabled_is_refused(self) -> None:
        """Otherwise a profile that collects nothing looks like a quiet day."""
        entity = dict(minimal()["watchlist"][0], enabled=False)  # type: ignore[index]
        with pytest.raises(ConfigurationError, match="no enabled watch entities"):
            CollectionProfile.from_mapping(minimal(watchlist=[entity]))

    def test_missing_keys_are_named(self) -> None:
        with pytest.raises(ConfigurationError, match="feeds"):
            CollectionProfile.from_mapping({"name": "x", "watchlist": []})

    def test_a_missing_file_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            CollectionProfile.load(tmp_path / "absent.json")

    def test_malformed_json_says_so(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            CollectionProfile.load(broken)


# ------------------------------------------------------- credentials and leaks


class TestCredentialsAreNeverInvented:
    def test_a_licensed_provider_is_unavailable_without_its_credential(self) -> None:
        profile = CollectionProfile.from_mapping(minimal())
        report = {entry.provider_id: entry for entry in profile.availability({})}
        assert report[NEWS_API_DECLARATION.provider_id].available is False
        assert NEWS_API_DECLARATION.credentials_env in report[NEWS_API_DECLARATION.provider_id].reason

    def test_naming_a_credential_is_not_enabling_the_provider(self) -> None:
        """The variable being set says a contract may exist. It does not say this
        profile is allowed to use it, so the provider stays off."""
        profile = CollectionProfile.from_mapping(minimal())
        environment = {str(NEWS_API_DECLARATION.credentials_env): "a-real-looking-key"}
        report = {entry.provider_id: entry for entry in profile.availability(environment)}
        assert report[NEWS_API_DECLARATION.provider_id].available is False
        assert "does not enable" in report[NEWS_API_DECLARATION.provider_id].reason

    def test_public_feeds_are_available_with_no_credential_at_all(self) -> None:
        profile = CollectionProfile.from_mapping(minimal())
        report = {entry.provider_id: entry for entry in profile.availability({})}
        assert report[SYNDICATION].available is True

    def test_no_credential_value_reaches_the_manifest_or_the_report(self) -> None:
        """The fingerprint is written into the corpus and read by whoever
        inherits it. A secret in there is permanent."""
        secret = "sk-live-do-not-log-me"
        profile = CollectionProfile.from_mapping(minimal(user_agent=f"a/1 ({secret}@x.org)"))
        environment = {str(NEWS_API_DECLARATION.credentials_env): secret}
        rendered = json.dumps(profile.fingerprint()) + json.dumps(
            [entry.as_dict() for entry in profile.availability(environment)]
        )
        assert secret not in rendered

    def test_the_user_agent_itself_is_not_published_in_the_fingerprint(self) -> None:
        profile = CollectionProfile.from_mapping(minimal(user_agent="a/1 (ops@example.org)"))
        assert profile.fingerprint()["contact_user_agent_configured"] is True
        assert "ops@example.org" not in json.dumps(profile.fingerprint())


# ------------------------------------------------------------------- retrieval


class TestRetrieverConstruction:
    def test_a_provider_that_is_not_due_is_never_constructed(self) -> None:
        profile = CollectionProfile.from_mapping(minimal())
        assert profile.build_retriever([]).providers == {}

    def test_a_due_provider_carries_the_configured_user_agent(self) -> None:
        agent = "btc-intel/1.0 (ops@example.org)"
        profile = CollectionProfile.from_mapping(minimal(user_agent=agent))
        provider = profile.build_retriever([SYNDICATION]).providers[SYNDICATION]
        assert provider.user_agent == agent  # type: ignore[attr-defined]

    def test_an_unconfigured_user_agent_falls_back_without_claiming_a_contact(self) -> None:
        profile = CollectionProfile.from_mapping(minimal())
        provider = profile.build_retriever([SYNDICATION]).providers[SYNDICATION]
        assert provider.user_agent == syndication.DEFAULT_USER_AGENT  # type: ignore[attr-defined]
        assert "@" not in syndication.DEFAULT_USER_AGENT, "the default must not fake a contact"

    def test_the_user_agent_reaches_the_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of the field is the header. Assert the header."""
        seen: list[str] = []

        class Recorder:
            def __init__(self, url: str, headers: dict[str, str]) -> None:
                seen.append(headers["User-Agent"])

        class Response:
            def read(self) -> bytes:
                return FEED_PAYLOAD

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        monkeypatch.setattr(syndication.urllib.request, "Request", Recorder)
        monkeypatch.setattr(syndication.urllib.request, "urlopen", lambda *a, **k: Response())

        agent = "btc-intel/1.0 (ops@example.org)"
        profile = CollectionProfile.from_mapping(minimal(user_agent=agent))
        provider = profile.build_retriever([SYNDICATION]).providers[SYNDICATION]
        provider.search("bitcoin regulation", NOW - timedelta(days=1), NOW)
        assert seen == [agent]


# ----------------------------------------------------------- one whole cycle


class TestScheduledCycleFromAProfile:
    """End to end through the deployed path, with the network replaced."""

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            syndication, "_default_opener", lambda url, timeout, agent=None: FEED_PAYLOAD
        )

    def test_a_cycle_collects_and_writes_its_manifest_last(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        outcome = collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        assert outcome.exit_code == EXIT_OK
        assert outcome.ran
        assert outcome.result is not None and outcome.result.manifest.documents_new >= 1
        assert outcome.manifest_path is not None and outcome.manifest_path.exists()
        assert not paths.lock.exists()

    def test_the_fingerprint_in_the_manifest_can_be_resolved_back(self, tmp_path: Path) -> None:
        """The manifest carries a hash of the configuration, which proves two
        runs matched and says nothing about what either ran with. Keeping the
        preimage is what separates "the feed was off" from "the feed was quiet"
        six months later."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        outcome = collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        assert outcome.result is not None
        fingerprint = outcome.result.manifest.configuration_fingerprint
        preimage = paths.manifests / f"configuration-{fingerprint}.json"
        assert preimage.exists(), "the manifest's fingerprint resolves to nothing"
        recorded = json.loads(preimage.read_text(encoding="utf-8"))
        assert recorded["configuration"]["feeds"] == ["sec-press"]

    def test_a_stable_configuration_records_itself_once(self, tmp_path: Path) -> None:
        """Content-addressed: months of cycles must not accumulate months of
        identical copies."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)
        collect_once(paths, profile, now=lambda: NOW + timedelta(hours=2), require_free_bytes=1)
        assert len(list(paths.manifests.glob("configuration-*.json"))) == 1

    def test_changing_the_configuration_does_not_overwrite_the_old_record(
        self, tmp_path: Path
    ) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        collect_once(
            paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1
        )
        widened = CollectionProfile.from_mapping(minimal(feeds=["sec-press", "cftc-press"]))
        collect_once(paths, widened, now=lambda: NOW + timedelta(hours=2), require_free_bytes=1)

        records = sorted(paths.manifests.glob("configuration-*.json"))
        assert len(records) == 2, "the earlier configuration was overwritten"
        recorded = [json.loads(r.read_text(encoding="utf-8"))["configuration"]["feeds"] for r in records]
        assert ["sec-press"] in recorded and ["cftc-press", "sec-press"] in recorded

    def test_collection_is_point_in_time_by_retrieval(self, tmp_path: Path) -> None:
        """The feed says 09:30. Availability is when we retrieved it, not when
        the publisher says it happened."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        store = IntelligenceStore(paths.database)
        try:
            documents = store.documents_as_of(NOW + timedelta(minutes=1))
            assert documents, "the cycle collected nothing"
            for document in documents:
                assert document.available_at == NOW
                assert document.published_at is None or document.published_at <= document.available_at
        finally:
            store.close()

    def test_a_second_cycle_inside_the_floor_does_nothing(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)
        again = collect_once(
            paths, profile, now=lambda: NOW + timedelta(minutes=5), require_free_bytes=1
        )
        assert again.exit_code == EXIT_NOTHING_DUE
        assert not again.ran

    def test_rediscovery_does_not_move_availability(self, tmp_path: Path) -> None:
        """The defect that made a day-1 document invisible to a day-2 replay.
        It is worth re-asserting through the deployed path specifically."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        store = IntelligenceStore(paths.database)
        before = {d.document_id: d.available_at for d in store.documents_as_of(NOW)}
        store.close()

        later = NOW + timedelta(days=2)
        second = collect_once(paths, profile, now=lambda: later, require_free_bytes=1)
        assert second.ran

        store = IntelligenceStore(paths.database)
        try:
            after = {
                d.document_id: d.available_at for d in store.documents_as_of(later + timedelta(hours=1))
            }
        finally:
            store.close()
        assert before, "nothing was collected, so nothing was proven"
        assert {k: after[k] for k in before} == before

    def test_a_held_lock_is_reported_not_failed(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        holder = RunLock(paths.lock, now=lambda: NOW).acquire()
        try:
            outcome = collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)
            assert outcome.exit_code == EXIT_LOCK_HELD
            assert not outcome.ran
        finally:
            holder.release()


# ------------------------------------------------------------------- the CLI


class TestSchedulerEntrypoint:
    """What a unit file actually runs, exercised as a scheduler would."""

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            syndication, "_default_opener", lambda url, timeout, agent=None: FEED_PAYLOAD
        )

    def _profile_file(self, tmp_path: Path) -> Path:
        target = tmp_path / "profile.json"
        target.write_text(json.dumps(minimal()), encoding="utf-8")
        return target

    def test_exit_codes_are_the_whole_interface(self, tmp_path: Path) -> None:
        """0 ran, 4 nothing due. A scheduler that treats 4 as failure pages
        someone every night for a collector behaving exactly as designed."""
        from market_intelligence.cli import main

        profile = self._profile_file(tmp_path)
        root = str(tmp_path / "state")
        assert main(["collect-scheduled", "--profile", str(profile), "--state-root", root]) == EXIT_OK
        assert (
            main(["collect-scheduled", "--profile", str(profile), "--state-root", root])
            == EXIT_NOTHING_DUE
        )

    def test_every_run_leaves_a_log_behind(self, tmp_path: Path) -> None:
        """A scheduler keeps the last few runs of stdout and the corpus keeps
        manifests. Neither records what the collector decided and why."""
        from market_intelligence.cli import main

        root = tmp_path / "state"
        main(
            [
                "collect-scheduled",
                "--profile",
                str(self._profile_file(tmp_path)),
                "--state-root",
                str(root),
                "--json",
            ]
        )
        logs = list((root / "logs").glob("collect-*.json"))
        assert len(logs) == 1
        recorded = json.loads(logs[0].read_text(encoding="utf-8"))
        assert recorded["exit_code"] == EXIT_OK
        assert recorded["profile"]["feeds"] == ["sec-press"]

    def test_a_bad_profile_fails_before_touching_the_corpus(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exit 2 and one sentence. A traceback escaping to the shell breaks the
        documented interface twice: it exits 1, which the contract does not
        mention, and it hands an operator fifteen lines of Python at 3am."""
        from market_intelligence.cli import main
        from market_intelligence.ops.scheduled import EXIT_FAILED

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(minimal(feeds=["not-a-feed"])), encoding="utf-8")
        code = main(
            ["collect-scheduled", "--profile", str(bad), "--state-root", str(tmp_path / "s")]
        )
        assert code == EXIT_FAILED
        assert "Traceback" not in capsys.readouterr().err
        assert not (tmp_path / "s" / "logs").exists()

    def test_a_missing_profile_is_a_documented_failure_not_an_undocumented_one(
        self, tmp_path: Path
    ) -> None:
        from market_intelligence.cli import main
        from market_intelligence.ops.scheduled import EXIT_FAILED

        assert (
            main(
                [
                    "collect-scheduled",
                    "--profile",
                    str(tmp_path / "absent.json"),
                    "--state-root",
                    str(tmp_path / "s"),
                ]
            )
            == EXIT_FAILED
        )

    def test_the_lock_is_reported_through_the_exit_code(self, tmp_path: Path) -> None:
        from market_intelligence.cli import main

        root = tmp_path / "state"
        paths = StoragePaths.from_environment(root).ensure()
        holder = RunLock(paths.lock).acquire()
        try:
            code = main(
                [
                    "collect-scheduled",
                    "--profile",
                    str(self._profile_file(tmp_path)),
                    "--state-root",
                    str(root),
                ]
            )
            assert code == EXIT_LOCK_HELD
        finally:
            holder.release()


# ------------------------------------------------------------- the unit files


DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
UNITS = DEPLOY / "systemd"


class TestUnitFilesMatchTheCode:
    """Unit files rot silently: nothing recompiles them, and a wrong one fails
    at 03:07 on a host nobody is watching."""

    def _unit(self, name: str) -> str:
        return (UNITS / name).read_text(encoding="utf-8")

    def test_the_collector_unit_invokes_a_command_that_exists(self) -> None:
        from market_intelligence.cli import build_parser

        unit = self._unit("btc-intel-collect.service")
        assert "collect-scheduled" in unit
        # argparse raises SystemExit on an unknown subcommand, so this is a
        # genuine check that the unit names a real command with real flags.
        parsed = build_parser().parse_args(
            ["collect-scheduled", "--profile", "p.json", "--json"]
        )
        assert parsed.command == "collect-scheduled"

    def test_the_unit_accepts_exactly_the_non_failure_exit_codes(self) -> None:
        """3 is 'another cycle is running' and 4 is 'nothing due'. If either
        counted as a failure the operator would be paged nightly, and would stop
        looking at the one that mattered."""
        unit = self._unit("btc-intel-collect.service")
        declared = [
            line.split("=", 1)[1].split()
            for line in unit.splitlines()
            if line.startswith("SuccessExitStatus=")
        ]
        assert declared, "the unit does not tolerate the collector's normal exit codes"
        assert set(declared[0]) == {str(EXIT_LOCK_HELD), str(EXIT_NOTHING_DUE)}

    def test_failure_is_still_failure(self) -> None:
        from market_intelligence.ops.scheduled import EXIT_FAILED

        declared = next(
            line
            for line in self._unit("btc-intel-collect.service").splitlines()
            if line.startswith("SuccessExitStatus=")
        )
        assert str(EXIT_FAILED) not in declared.split("=", 1)[1].split()

    def test_every_timer_survives_a_host_being_off(self) -> None:
        """Without Persistent, a reboot at the wrong minute silently drops a
        collection, and the corpus records a gap that reads as a quiet day."""
        for timer in UNITS.glob("*.timer"):
            assert "Persistent=true" in timer.read_text(encoding="utf-8"), timer.name

    def test_the_collection_cadence_respects_the_declared_floor(self) -> None:
        unit = self._unit("btc-intel-collect.timer")
        calendar = next(
            line for line in unit.splitlines() if line.startswith("OnCalendar=")
        )
        hours = int(calendar.split("00/")[1].split(":")[0])
        assert hours * 3600 >= SYNDICATION_DECLARATION.minimum_interval_seconds

    def test_units_point_at_the_committed_profile(self) -> None:
        assert "collection-profile.json" in self._unit("btc-intel-collect.service")
        assert (DEPLOY / "collection-profile.json").exists()

    def test_no_unit_or_example_carries_a_credential(self) -> None:
        """The deployment needs no secret at all, and that is worth keeping:
        a collector with no credential cannot leak one."""
        for path in list(UNITS.iterdir()) + [DEPLOY / "collector.env.example"]:
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                bare = line.strip()
                if bare.startswith("#") or "=" not in bare:
                    continue
                key, _, value = bare.partition("=")
                if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    assert not value.strip(), f"{path.name} ships a value for {key}"

    def test_the_installer_never_deletes_at_the_destination(self) -> None:
        """An rsync --delete into a prefix an operator has put state inside is
        how a deployment eats its own corpus."""
        installer = (DEPLOY / "install.sh").read_text(encoding="utf-8")
        invocations = [line for line in installer.splitlines() if line.strip().startswith("rsync")]
        assert invocations, "the installer no longer copies code the way this test assumes"
        assert not any("--delete" in line for line in invocations)

    def test_the_installer_does_not_overwrite_an_existing_env_file(self) -> None:
        installer = (DEPLOY / "install.sh").read_text(encoding="utf-8")
        assert "left alone" in installer

    def test_the_state_root_is_not_ephemeral(self) -> None:
        """systemd's StateDirectory lands under /var/lib and survives reboots
        and package updates; /tmp does not."""
        from market_intelligence.ops.paths import looks_ephemeral

        unit = self._unit("btc-intel-collect.service")
        assert "StateDirectory=btc-intel" in unit
        assert looks_ephemeral(Path("/var/lib/btc-intel")) is None


# ------------------------------------------- what the operator is told happened


class TestTheCycleReportsWhatActuallyHappened:
    """A live run reported twelve quarantined records against an empty
    quarantine table. The number was the deduplication count, which is the
    normal shape of a cycle: the same SEC release matches several queries. An
    operator who reads that either investigates a non-problem or learns the
    field is noise, and the second is how a real quarantine goes unnoticed."""

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            syndication, "_default_opener", lambda url, timeout, agent=None: FEED_PAYLOAD
        )

    def test_duplicates_are_reported_as_duplicates(self, tmp_path: Path) -> None:
        """Several watch entities against one feed: the same item arrives more
        than once, and none of it is a fault."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        # Two topics that both match the one item, which is what a real
        # watchlist does constantly: one release answers several queries.
        watchlist = [
            dict(  # type: ignore[index]
                minimal()["watchlist"][0],
                topics=["bitcoin regulation", "regulation decision"],
            )
        ]
        profile = CollectionProfile.from_mapping(minimal(watchlist=watchlist))
        outcome = collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        assert outcome.result is not None
        manifest = outcome.result.manifest
        assert manifest.documents_deduplicated >= 1, "this feed did not duplicate; test is vacuous"
        assert manifest.quarantined == 0, "deduplication is not quarantine"

    def test_a_failing_provider_is_reported_as_quarantine(self, tmp_path: Path) -> None:
        from market_intelligence.collection.syndication import ProviderFailure

        def broken(url: str, timeout: float, agent: object = None) -> bytes:
            raise ProviderFailure("PERMANENT", f"HTTP 403 for {url}")

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(syndication, "_default_opener", broken)
            outcome = collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        assert outcome.result is not None
        assert outcome.result.manifest.quarantined >= 1
        assert outcome.result.manifest.documents_deduplicated == 0

    def test_the_watchdog_is_given_the_real_quarantine_count(self, tmp_path: Path) -> None:
        """It defaulted to zero on the deployed path, so the quarantine alert
        could never fire however bad collection got."""
        from market_intelligence.cli import _ops_report

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        collect_once(
            paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1
        )

        store = IntelligenceStore(paths.database)
        try:
            from market_intelligence.operations import QuarantineRecord

            store.put_quarantine(
                [
                    QuarantineRecord.from_raw(
                        f"failure {index}",
                        "syndication",
                        datetime.now(timezone.utc),
                        "boom",
                        "PROVIDER_FAILURE",
                    )
                    for index in range(40)
                ]
            )
            report = _ops_report(store, paths.database, str(paths.root))
        finally:
            store.close()

        spikes = [
            alert
            for alert in report["watchdog"]["alerts"]
            if alert["code"] == "QUARANTINE_SPIKE"
        ]
        assert spikes, "the quarantine alert still cannot fire"
        assert spikes[0]["detail"]["quarantined"] == 40

    def test_an_old_quarantine_backlog_does_not_latch_the_alert_on(
        self, tmp_path: Path
    ) -> None:
        """A spike is about a window. A historical total would keep the alert
        raised forever and make it worthless."""
        from market_intelligence.operations import QuarantineRecord

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        collect_once(
            paths, CollectionProfile.from_mapping(minimal()), now=lambda: NOW, require_free_bytes=1
        )
        store = IntelligenceStore(paths.database)
        try:
            ancient = datetime.now(timezone.utc) - timedelta(days=30)
            store.put_quarantine(
                [
                    QuarantineRecord.from_raw(f"old {i}", "syndication", ancient, "boom", "PROVIDER_FAILURE")
                    for i in range(40)
                ]
            )
            assert store.quarantine_count() == 40
            assert store.quarantine_count(since=datetime.now(timezone.utc) - timedelta(hours=24)) == 0
        finally:
            store.close()


# ----------------------------------------------- events under repeated cycles


class TestRediscoveryDoesNotManufactureEvents:
    """The document layer had first-write-wins from B4.1. The event layer did
    not read it: extraction ran on the freshly retrieved objects, whose
    `available_at` is this cycle's retrieval time, so an unchanged document
    produced a new event every cycle.

    At the deployed three-hour cadence a press release that sits in a feed for a
    week becomes fifty-six events. B4's readiness gate counts events, so it
    would have opened on duplicates of a handful of announcements -- reporting a
    corpus ready for study when it held almost nothing. The mirror image of the
    publisher-count defect, and the more dangerous direction of the two."""

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            syndication, "_default_opener", lambda url, timeout, agent=None: FEED_PAYLOAD
        )

    def _events(self, paths: StoragePaths, when: datetime) -> list[object]:
        store = IntelligenceStore(paths.database)
        try:
            return list(store.signals_as_of(when))
        finally:
            store.close()

    def test_an_unchanged_document_yields_the_same_event(self, tmp_path: Path) -> None:
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())

        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)
        first = self._events(paths, NOW + timedelta(minutes=1))
        assert first, "nothing was extracted, so nothing is being proven"

        later = NOW + timedelta(hours=4)
        collect_once(paths, profile, now=lambda: later, require_free_bytes=1)
        second = self._events(paths, later + timedelta(minutes=1))

        assert len(second) == len(first), "rediscovery manufactured events"
        assert {e.event_id for e in second} == {e.event_id for e in first}  # type: ignore[attr-defined]

    def test_event_availability_does_not_drift_forward(self, tmp_path: Path) -> None:
        """An event whose availability moves with the last time we happened to
        look is not point-in-time evidence about anything."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())

        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)
        before = {
            e.event_id: e.available_time  # type: ignore[attr-defined]
            for e in self._events(paths, NOW + timedelta(minutes=1))
        }

        later = NOW + timedelta(days=3)
        collect_once(paths, profile, now=lambda: later, require_free_bytes=1)
        after = {
            e.event_id: e.available_time  # type: ignore[attr-defined]
            for e in self._events(paths, later + timedelta(minutes=1))
        }
        assert after == before

    def test_many_cycles_do_not_inflate_the_corpus(self, tmp_path: Path) -> None:
        """Eight cycles is one day at the deployed cadence."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())

        for cycle in range(8):
            collect_once(
                paths, profile, now=lambda c=cycle: NOW + timedelta(hours=3 * c), require_free_bytes=1
            )

        horizon = NOW + timedelta(days=2)
        store = IntelligenceStore(paths.database)
        try:
            documents = store.documents_as_of(horizon)
            events = store.signals_as_of(horizon)
        finally:
            store.close()

        assert len(documents) == 1, "the fixture feed serves one item"
        assert len(events) <= len(documents), f"{len(events)} events from {len(documents)} documents"

    def test_the_readiness_gate_is_not_opened_by_repetition(self, tmp_path: Path) -> None:
        """The consequence that actually matters: 30 events is the bar, and one
        announcement collected for four days used to clear it."""
        from market_intelligence.collection.readiness import assess_family

        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        for cycle in range(32):
            collect_once(
                paths, profile, now=lambda c=cycle: NOW + timedelta(hours=3 * c), require_free_bytes=1
            )

        store = IntelligenceStore(paths.database)
        try:
            events = store.signals_as_of(NOW + timedelta(days=10))
            documents = store.documents_as_of(NOW + timedelta(days=10))
        finally:
            store.close()

        from market_intelligence.collection.clustering import cluster_events

        clusters = cluster_events(list(events), list(documents), window_hours=24)
        assert not assess_family("regulation", clusters).ready

    def test_a_genuinely_new_document_still_produces_an_event(self, tmp_path: Path) -> None:
        """The fix must not be 'stop extracting'."""
        paths = StoragePaths.from_environment(tmp_path / "state").ensure()
        profile = CollectionProfile.from_mapping(minimal())
        collect_once(paths, profile, now=lambda: NOW, require_free_bytes=1)

        second_item = FEED_PAYLOAD.replace(
            b"https://sec.example.gov/news/2026/decision",
            b"https://sec.example.gov/news/2026/second",
        ).replace(b"announces bitcoin regulation decision", b"issues bitcoin regulation guidance")

        later = NOW + timedelta(hours=4)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                syndication, "_default_opener", lambda url, timeout, agent=None: second_item
            )
            collect_once(paths, profile, now=lambda: later, require_free_bytes=1)

        store = IntelligenceStore(paths.database)
        try:
            assert len(store.documents_as_of(later + timedelta(minutes=1))) == 2
            assert len(store.signals_as_of(later + timedelta(minutes=1))) == 2
        finally:
            store.close()
