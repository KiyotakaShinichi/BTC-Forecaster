"""B5.2 -- how an operator hears that the collector has stopped.

A production collector that fails quietly is the failure this whole deployment
exists to avoid: a gap in the corpus reads, afterwards, exactly like a quiet
week. So every unit that can fail -- a cycle, the integrity check, the backup,
the health check -- names `btc-intel-alert@.service` as its OnFailure, and that
unit runs `btc-intel ops-alert --unit <name>`.

The alert always goes to the journal, at CRITICAL, as one structured record.
Where it goes beyond that is the operator's decision, not this repository's:
`BTC_INTEL_ALERT_COMMAND`, if set, is run with the message on its standard
input -- `mail -s "btc-intel alert" ops@example.org`, a webhook `curl`, whatever
the host already has. No mail server, no paging service and no dashboard is
assumed or installed.

The command may carry a token, so its text is never printed or logged, and its
value is scrubbed from every log line like any other secret.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..logs import get_logger

#: Optional. Run through the shell with the alert message on stdin.
ALERT_COMMAND_ENV = "BTC_INTEL_ALERT_COMMAND"

#: An alert command that hangs must not hang the alert unit forever.
ALERT_COMMAND_TIMEOUT_SECONDS = 60

_log = get_logger("alert")


@dataclass(frozen=True)
class AlertOutcome:
    unit: str
    message: str
    command_configured: bool
    command_exit_code: int | None

    @property
    def delivered(self) -> bool:
        """Written to the journal always; delivered beyond it if a command ran cleanly."""
        return not self.command_configured or self.command_exit_code == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "message": self.message,
            "command_configured": self.command_configured,
            "command_exit_code": self.command_exit_code,
            "delivered": self.delivered,
        }


def alert_message(unit: str, host: str, moment: datetime) -> str:
    stamp = moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"btc-intel ALERT: {unit} failed on {host} at {stamp}. "
        f"Read `journalctl -u {unit} -n 50` and `btc-intel ops-status` on the host."
    )


def raise_alert(
    unit: str,
    *,
    environment: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> AlertOutcome:
    """Record the alert, and hand it to the operator's command if there is one."""
    env = os.environ if environment is None else environment
    message = alert_message(unit, socket.gethostname(), now())
    _log.emit("alert_raised", severity=logging.CRITICAL, unit=unit, message=message)

    command = (env.get(ALERT_COMMAND_ENV) or "").strip()
    if not command:
        return AlertOutcome(unit, message, False, None)
    try:
        result = run(
            command,
            shell=True,  # noqa: S602 -- the operator's own configured command, from a root-owned file
            input=message,
            text=True,
            capture_output=True,
            timeout=ALERT_COMMAND_TIMEOUT_SECONDS,
        )
        exit_code = int(result.returncode)
    except (OSError, subprocess.TimeoutExpired) as error:
        _log.emit("alert_command_failed", severity=logging.ERROR, unit=unit, detail=type(error).__name__)
        return AlertOutcome(unit, message, True, -1)
    if exit_code != 0:
        # The exit code only. The command's own output may echo its arguments.
        _log.emit("alert_command_failed", severity=logging.ERROR, unit=unit, exit_code=exit_code)
    return AlertOutcome(unit, message, True, exit_code)


__all__ = ["ALERT_COMMAND_ENV", "AlertOutcome", "alert_message", "raise_alert"]
