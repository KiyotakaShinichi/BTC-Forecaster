"""B5.2 — the production service: a collector that is watched and says when it stops.

B4.1-Ops shipped three timers -- collect, verify, backup -- and a watchdog
command nothing ran. A unit that failed failed silently, and a failed backup was
noticed, if at all, when the archive turned out to be a week old.

Pinned here:

* every unit that can fail raises btc-intel-alert@.service, and the alert unit
  raises nothing about itself;
* the health check runs hourly, clear of the collection minute; a warning is
  logged and a critical fails the unit;
* the backup unit is ops-backup, which verifies before it prunes; a held lock
  is not a failure;
* every unit invokes a command and flags that genuinely parse;
* the installer checks the configuration before it enables any timer, and
  enables the health check too;
* `ops-alert` writes the alert to the log always, runs the operator's command
  with the message on standard input when there is one, and never prints the
  command.

No network, no systemd: the unit files are read as text and their commands
parsed by the real parser.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from market_intelligence.cli import build_parser, main
from market_intelligence.ops.alert import ALERT_COMMAND_ENV, raise_alert

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
UNITS = DEPLOY / "systemd"
FAILING_UNITS = ("btc-intel-collect.service", "btc-intel-verify.service", "btc-intel-backup.service", "btc-intel-watch.service")


def unit(name: str) -> str:
    return (UNITS / name).read_text(encoding="utf-8")


def directive(text: str, key: str) -> list[str]:
    return [line.split("=", 1)[1] for line in text.splitlines() if line.startswith(f"{key}=")]


def invoked(text: str) -> list[str]:
    """The btc-intel arguments a unit's ExecStart passes, with systemd specifiers made concrete."""
    (command,) = directive(text, "ExecStart")
    words = shlex.split(command.replace("%S", "/var/lib").replace("%i", "btc-intel-collect.service"))
    return words[words.index("/opt/btc-intel/btc-intel.py") + 1 :]


class TestAlertsAreWiredEverywhere:
    @pytest.mark.parametrize("name", FAILING_UNITS)
    def test_every_unit_that_can_fail_raises_an_alert(self, name: str) -> None:
        assert directive(unit(name), "OnFailure") == ["btc-intel-alert@%n.service"]

    def test_the_alert_unit_raises_nothing_about_itself(self) -> None:
        assert directive(unit("btc-intel-alert@.service"), "OnFailure") == []

    def test_the_alert_unit_runs_ops_alert_for_the_failed_unit(self) -> None:
        arguments = invoked(unit("btc-intel-alert@.service"))
        parsed = build_parser().parse_args(arguments)
        assert (parsed.command, parsed.unit) == ("ops-alert", "btc-intel-collect.service")


class TestTheHealthCheckRuns:
    def test_hourly_and_clear_of_the_collection_minute(self) -> None:
        (calendar,) = directive(unit("btc-intel-watch.timer"), "OnCalendar")
        minute = calendar.split(":")[1]
        (collect,) = directive(unit("btc-intel-collect.timer"), "OnCalendar")
        assert calendar == "*-*-* *:37:00" and minute != collect.split(":")[1]

    def test_it_checks_the_deployed_profile(self) -> None:
        parsed = build_parser().parse_args(invoked(unit("btc-intel-watch.service")))
        assert parsed.command == "ops-watch" and parsed.profile.endswith("deploy/collection-profile.json")
        assert "EnvironmentFile=-/etc/btc-intel/collector.env" in unit("btc-intel-watch.service")

    def test_a_warning_is_logged_and_a_critical_fails(self) -> None:
        assert directive(unit("btc-intel-watch.service"), "SuccessExitStatus") == ["1"]


class TestTheBackupUnit:
    def test_it_runs_the_verifying_backup(self) -> None:
        parsed = build_parser().parse_args(invoked(unit("btc-intel-backup.service")))
        assert parsed.command == "ops-backup" and parsed.retain >= 1 and parsed.wait_seconds > 0

    def test_a_held_lock_is_not_a_failure_but_a_failed_backup_is(self) -> None:
        assert directive(unit("btc-intel-backup.service"), "SuccessExitStatus") == ["3"]


class TestEveryUnitIsRealAndHardened:
    @pytest.mark.parametrize("name", sorted(path.name for path in UNITS.glob("*.service")))
    def test_its_command_parses(self, name: str) -> None:
        build_parser().parse_args(invoked(unit(name)))

    @pytest.mark.parametrize("name", sorted(path.name for path in UNITS.glob("*.service")))
    def test_it_runs_unprivileged_on_a_read_only_system(self, name: str) -> None:
        text = unit(name)
        for required in ("User=btc-intel", "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes"):
            assert required in text, (name, required)

    def test_only_the_collector_and_the_alert_may_reach_the_network(self) -> None:
        for path in UNITS.glob("*.service"):
            families = directive(path.read_text(encoding="utf-8"), "RestrictAddressFamilies")
            networked = any("AF_INET" in value for value in families)
            assert networked == (path.name in {"btc-intel-collect.service", "btc-intel-alert@.service"}), path.name


class TestTheInstaller:
    INSTALLER = DEPLOY / "install.sh"

    def test_it_checks_the_configuration_before_enabling_any_timer(self) -> None:
        text = self.INSTALLER.read_text(encoding="utf-8")
        assert "ops-config-check" in text
        assert text.index("ops-config-check") < text.index("systemctl enable --now")

    def test_it_enables_all_four_timers(self) -> None:
        text = self.INSTALLER.read_text(encoding="utf-8")
        for timer in UNITS.glob("*.timer"):
            assert f"systemctl enable --now {timer.name}" in text, timer.name

    @pytest.mark.skipif(
        os.name == "nt" or shutil.which("bash") is None,
        reason="needs a POSIX bash: on Windows `bash` may be WSL's, which cannot read a Windows path. CI runs this, and quality.yml runs `bash -n` too",
    )
    def test_it_parses(self) -> None:
        result = subprocess.run(["bash", "-n", str(self.INSTALLER)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestTheAlert:
    def test_with_no_command_it_is_written_to_the_log_and_delivered(self) -> None:
        outcome = raise_alert("btc-intel-collect.service", environment={})
        assert outcome.delivered and not outcome.command_configured
        assert "btc-intel-collect.service failed" in outcome.message

    def test_the_operators_command_receives_the_message(self, tmp_path: Path) -> None:
        received = tmp_path / "received.txt"
        command = f'"{sys.executable}" -c "import sys; open(r\'{received}\', \'w\').write(sys.stdin.read())"'
        outcome = raise_alert("btc-intel-backup.service", environment={ALERT_COMMAND_ENV: command})
        assert outcome.delivered and outcome.command_exit_code == 0
        assert received.read_text() == outcome.message

    def test_a_failing_command_is_reported_without_its_text(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        secret = "https://hooks.example.org/t0k3n-s3cr3t"
        monkeypatch.setenv(ALERT_COMMAND_ENV, f'"{sys.executable}" -c "import sys; sys.exit(7)" {secret}')
        code = main(["ops-alert", "--unit", "btc-intel-watch.service"])
        captured = capsys.readouterr()
        assert code == 2
        assert json.loads(captured.out)["command_exit_code"] == 7
        assert secret not in captured.out + captured.err
