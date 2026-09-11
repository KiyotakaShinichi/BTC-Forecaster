"""Structured operational logging, and somewhere for it to go.

There was already a structured logger here. `StructuredRunLogger` emitted JSON
through `logging` and dropped fields whose names looked like credentials, and
two of the three surfaces that logged anything used it. What it did not have
was a handler: nothing in this repository ever called `basicConfig` or attached
one, so every `INFO` record fell through to the root logger's last-resort
handler at `WARNING` and was discarded. The logs were structured and nobody
could read them.

So this module does three things and adds no framework:

* it moves the emitter out of `cycle.py`, where the replay path happened to
  own it, so the collection path and the API can use the same one;
* it configures a destination -- one JSON object per line on stderr, which is
  what systemd's journal wants and what `journalctl -o cat | jq` reads;
* it redacts by value as well as by name, so a secret cannot reach a log by
  travelling inside a field that is not called `api_key`.

**Field names are a contract.** `run_id`, `provider_id`, `origin`, `event` and
`severity` mean the same thing everywhere they appear, because the whole reason
to emit JSON is that something other than a human reads it.

Configuration is explicit and belongs to entry points. A library that installs
a handler on import steals the destination from whatever imported it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import IO, Any, Mapping

#: One logger tree. Everything under it is this system's.
LOGGER_NAME = "btc_intelligence"

#: `BTC_INTEL_LOG_LEVEL=DEBUG` on a host, without editing a unit file.
LEVEL_ENV = "BTC_INTEL_LOG_LEVEL"

#: Field names that may never carry a value into a log line, whatever they hold.
#: `user_agent` is here because the collector's contains the operator's contact
#: address -- the one value the deployment was explicitly told never to commit.
SENSITIVE_NAME_FRAGMENTS = (
    "credential",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "contact",
    "user_agent",
)

#: Environment variables whose *values* are scrubbed wherever they appear. Name
#: filtering alone is not enough: a contact address reaches a log through
#: `detail`, `reason` or an exception message far more easily than through a
#: field somebody named `contact`.
SECRET_ENV_VARIABLES = (
    "BTC_INTEL_CONTACT",
    # An alert command can carry a webhook token in its text.
    "BTC_INTEL_ALERT_COMMAND",
    "BTC_INTEL_SEARCH_API_KEY",
    "BTC_INTEL_STATEMENTS_API_KEY",
    "BTC_INTEL_WHALE_API_KEY",
)

REDACTED = "<redacted>"


def _secret_values(environment: Mapping[str, str] | None = None) -> tuple[str, ...]:
    env = os.environ if environment is None else environment
    values = {(env.get(name) or "").strip() for name in SECRET_ENV_VARIABLES}
    # A one-character "secret" would scrub half the alphabet out of every line.
    return tuple(sorted(value for value in values if len(value) > 3))


def scrub(value: object, secrets: tuple[str, ...]) -> object:
    """Replace any configured secret found inside a value.

    Recurses into containers, because a provider report is a list of dicts and
    an exception's `args` is a tuple.
    """
    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, Mapping):
        return {key: scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item, secrets) for item in value]
    return value


def redact(
    fields: Mapping[str, object], *, environment: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Drop fields named like a secret, and scrub secret values from the rest."""
    secrets = _secret_values(environment)
    return {
        key: scrub(value, secrets)
        for key, value in fields.items()
        if not any(fragment in key.casefold() for fragment in SENSITIVE_NAME_FRAGMENTS)
    }


class StructuredLogger:
    """Emits one JSON object per record, under a named logger.

    `emit` names its five contract fields explicitly rather than leaving them
    to `**fields`, so a caller who spells `runid` gets it as an ordinary field
    and not as the correlation key something downstream is grouping by.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(LOGGER_NAME)

    def emit(
        self,
        event: str,
        *,
        severity: int = logging.INFO,
        run_id: str | None = None,
        provider_id: str | None = None,
        origin: datetime | str | None = None,
        **fields: object,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "severity": logging.getLevelName(severity),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        if provider_id is not None:
            payload["provider_id"] = provider_id
        if origin is not None:
            payload["origin"] = origin.isoformat() if isinstance(origin, datetime) else origin
        payload.update(redact(fields))
        self.logger.log(severity, json.dumps(payload, default=str, sort_keys=True))


def get_logger(name: str | None = None) -> StructuredLogger:
    """A structured logger for one subsystem: `get_logger("api")`."""
    full = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    return StructuredLogger(logging.getLogger(full))


def configure(
    *,
    stream: IO[str] | None = None,
    level: int | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> logging.Logger:
    """Attach one JSON-lines handler to this system's logger tree.

    Idempotent: calling it twice does not double every line, which matters
    because a scheduler that invokes the CLI in a loop within one process would
    otherwise accumulate handlers.

    stderr rather than stdout on purpose. `btc-intel` commands write their
    result to stdout and a caller pipes it into `jq`; interleaving diagnostics
    there would corrupt the thing the caller actually asked for.
    """
    env = os.environ if environment is None else environment
    resolved = level if level is not None else env.get(LEVEL_ENV, "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)
    # Records stop here. Propagating to the root logger would print them a
    # second time under whatever format an embedding application had set up.
    logger.propagate = False

    destination = stream if stream is not None else sys.stderr
    for existing in list(logger.handlers):
        if getattr(existing, "_btc_intel_structured", False):
            logger.removeHandler(existing)

    handler = logging.StreamHandler(destination)
    # The record's message is already a complete JSON object; the formatter's
    # job is to not add anything to it.
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._btc_intel_structured = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


__all__ = [
    "LEVEL_ENV",
    "LOGGER_NAME",
    "REDACTED",
    "SECRET_ENV_VARIABLES",
    "SENSITIVE_NAME_FRAGMENTS",
    "StructuredLogger",
    "configure",
    "get_logger",
    "redact",
    "scrub",
]
