"""O2 — one collector at a time, and never wedged.

Two cron entries, or a manual run during a scheduled one, put two collectors on
one DuckDB file. The store's transaction protects individual writes, but two
cycles interleaving watermark advances and snapshot registration is not
something transactions make sensible.

The design constraint that matters more than exclusion: **a crash must not stop
collection forever.** A lock that survives the process holding it is worse than
no lock, because the failure is silent and permanent — collection simply stops,
and the watchdog reports staleness weeks later. So the lock carries the holder's
pid and a heartbeat, and a lock whose holder is demonstrably gone is broken
automatically and *reported*, never silently.

Deliberately not distributed. One host, one filesystem, `os.open` with
`O_EXCL` — which is atomic on POSIX and on Windows. A distributed lock would be
a second system to operate for a workload that has no second host.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from ..errors import IntelligenceError

#: A lock older than this with no live holder is stale. Generous on purpose: a
#: cycle that fetches a dozen feeds under retry can legitimately run for
#: minutes, and breaking a live lock is worse than waiting for a dead one.
DEFAULT_STALE_AFTER = timedelta(minutes=30)


class LockHeld(IntelligenceError):
    """Another collector holds the lock. Not an error to retry in a tight loop."""


@dataclass(frozen=True)
class LockRecord:
    """What the lock file says about its holder."""

    pid: int
    hostname: str
    acquired_at: datetime
    heartbeat_at: datetime
    purpose: str

    def as_json(self) -> str:
        return json.dumps(
            {
                "pid": self.pid,
                "hostname": self.hostname,
                "acquired_at": self.acquired_at.isoformat(),
                "heartbeat_at": self.heartbeat_at.isoformat(),
                "purpose": self.purpose,
            },
            sort_keys=True,
        )

    @classmethod
    def parse(cls, payload: str) -> "LockRecord | None":
        """A malformed lock file is treated as stale rather than fatal.

        Refusing to run because a lock file is corrupt would wedge collection on
        exactly the kind of partial write a crash produces -- the failure mode
        this lock exists to avoid.
        """
        try:
            record = json.loads(payload)
            return cls(
                pid=int(record["pid"]),
                hostname=str(record["hostname"]),
                acquired_at=datetime.fromisoformat(record["acquired_at"]),
                heartbeat_at=datetime.fromisoformat(record["heartbeat_at"]),
                purpose=str(record.get("purpose", "")),
            )
        except (ValueError, KeyError, TypeError):
            return None


def process_alive(pid: int) -> bool:
    """Whether a pid is running. Unknown counts as alive.

    Erring toward "alive" means a lock is broken only on positive evidence the
    holder is gone. The cost of being wrong that way is one skipped cycle; the
    cost of the other way is two collectors writing at once.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        # `os.kill(pid, 0)` is a liveness probe on POSIX, where the collector is
        # deployed. On Windows signal 0 is CTRL_C_EVENT: the call delivers Ctrl+C
        # to a console process group instead of asking about one, and reads a
        # dead pid as alive. There is no safe probe in the standard library, so
        # Windows answers "alive" and the heartbeat's stale window decides --
        # the same rule a lock held from another host already follows.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return True
    return True


class RunLock:
    """Filesystem lock with stale-holder recovery."""

    def __init__(
        self,
        path: str | Path,
        *,
        purpose: str = "collect",
        stale_after: timedelta = DEFAULT_STALE_AFTER,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        is_alive: Callable[[int], bool] = process_alive,
    ) -> None:
        self.path = Path(path)
        self.purpose = purpose
        self.stale_after = stale_after
        self._now = now
        self._is_alive = is_alive
        self.broke_stale_lock = False
        self.previous_holder: LockRecord | None = None
        self._acquired = False

    def read(self) -> LockRecord | None:
        if not self.path.exists():
            return None
        try:
            return LockRecord.parse(self.path.read_text(encoding="utf-8"))
        except OSError:
            return None

    def _is_stale(self, record: LockRecord | None) -> bool:
        if record is None:
            return True  # unreadable or malformed: see LockRecord.parse
        if record.hostname == _hostname() and not self._is_alive(record.pid):
            return True
        return self._now() - record.heartbeat_at > self.stale_after

    def acquire(self) -> "RunLock":
        """Take the lock, breaking a demonstrably dead one first."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = LockRecord(
            pid=os.getpid(),
            hostname=_hostname(),
            acquired_at=self._now(),
            heartbeat_at=self._now(),
            purpose=self.purpose,
        )
        try:
            self._create(record)
        except FileExistsError:
            existing = self.read()
            if not self._is_stale(existing):
                held_for = self._now() - existing.acquired_at if existing else None
                raise LockHeld(
                    f"{self.path} is held by pid {existing.pid if existing else '?'} on "
                    f"{existing.hostname if existing else '?'}"
                    + (f" for {held_for}" if held_for else "")
                ) from None
            # Stale: recorded, then broken. Silence here would hide a crash loop.
            self.previous_holder = existing
            self.broke_stale_lock = True
            self.path.unlink(missing_ok=True)
            self._create(record)
        self._acquired = True
        return self

    def _create(self, record: LockRecord) -> None:
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(descriptor, record.as_json().encode("utf-8"))
        finally:
            os.close(descriptor)

    def heartbeat(self) -> None:
        """Refresh the holder's timestamp during a long cycle."""
        if not self._acquired:
            return
        current = self.read()
        if current is None or current.pid != os.getpid():
            return
        refreshed = LockRecord(
            pid=current.pid,
            hostname=current.hostname,
            acquired_at=current.acquired_at,
            heartbeat_at=self._now(),
            purpose=current.purpose,
        )
        temporary = self.path.with_suffix(f"{self.path.suffix}.hb")
        temporary.write_text(refreshed.as_json(), encoding="utf-8")
        temporary.replace(self.path)

    def release(self) -> None:
        """Release only our own lock.

        Checking the pid matters: if this process's lock was broken as stale by
        another collector, the file now belongs to that one and deleting it
        would let a third in alongside it.
        """
        if not self._acquired:
            return
        current = self.read()
        if current is None or current.pid == os.getpid():
            self.path.unlink(missing_ok=True)
        self._acquired = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exception: object) -> None:
        self.release()


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - hostname lookup does not normally fail
        return "unknown"


def held_by(path: str | Path) -> LockRecord | None:
    """Inspect a lock without taking it, for the status command."""
    return RunLock(path).read()


def lock_iterator(lock: RunLock) -> Iterator[RunLock]:  # pragma: no cover - convenience
    with lock:
        yield lock


__all__ = [
    "DEFAULT_STALE_AFTER",
    "LockHeld",
    "LockRecord",
    "RunLock",
    "held_by",
    "process_alive",
]
