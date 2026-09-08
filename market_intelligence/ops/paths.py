"""O5 — every piece of state has a declared, validated, persistent home.

The failure this prevents is quiet and total: a container writes its DuckDB file
to a layer that is discarded on restart, collection appears to run perfectly for
weeks, and the corpus is empty every time anyone looks. Nothing errors. The
`available_at` semantics stay perfect. There is simply no accumulation.

So paths are declared in one place, resolved from environment variables with
documented defaults, and **validated at startup by actually writing to them**.
A directory that exists but is read-only, or is a full volume, or is a stale
mount, all pass an `exists()` check and fail on the first real write — hours
later, inside a cycle, where the failure looks like a provider problem.

`validate()` fails loudly. A collector that cannot persist should refuse to
start, not start and lose everything.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError

#: Environment variable prefix. One prefix, documented, so an operator can find
#: everything this system reads from the environment in one grep.
ENV_PREFIX = "BTC_INTEL_"

#: Default root. Deliberately not a temp directory: a default that silently
#: works and silently loses data is worse than one that requires a decision.
DEFAULT_ROOT = "./btc-intel-state"

#: The per-path override suffixes, in the order `from_environment` applies them.
#: Named here rather than only at the call sites so the environment surface can
#: be enumerated -- `.env.example` is checked against this, and a suffix added
#: without documenting it fails that check rather than staying invisible.
PATH_ENV_SUFFIXES: tuple[str, ...] = (
    "STATE_ROOT",
    "DATABASE",
    "MANIFEST_DIR",
    "CORPUS_DIR",
    "BACKUP_DIR",
    "LOG_DIR",
    "LOCK_FILE",
)

#: Every variable `StoragePaths` reads, fully qualified.
PATH_ENV_VARIABLES: tuple[str, ...] = tuple(
    f"{ENV_PREFIX}{suffix}" for suffix in PATH_ENV_SUFFIXES
)


@dataclass(frozen=True)
class StoragePaths:
    """Every path the collector needs, resolved and explicit."""

    root: Path
    database: Path
    manifests: Path
    corpus: Path
    backups: Path
    logs: Path
    lock: Path

    @classmethod
    def from_environment(cls, root: str | Path | None = None) -> "StoragePaths":
        """Resolve from `BTC_INTEL_STATE_ROOT`, or an explicit root.

        Individual paths can be overridden one at a time -- an operator with a
        fast disk for the database and a large one for backups should not have
        to choose between them.
        """
        configured = root or os.environ.get(f"{ENV_PREFIX}STATE_ROOT") or DEFAULT_ROOT
        base = Path(configured).expanduser()
        return cls(
            root=base,
            database=_override("DATABASE", base / "intelligence.duckdb"),
            manifests=_override("MANIFEST_DIR", base / "manifests"),
            corpus=_override("CORPUS_DIR", base / "corpus"),
            backups=_override("BACKUP_DIR", base / "backups"),
            logs=_override("LOG_DIR", base / "logs"),
            lock=_override("LOCK_FILE", base / "collector.lock"),
        )

    @property
    def directories(self) -> tuple[Path, ...]:
        return (self.root, self.manifests, self.corpus, self.backups, self.logs)

    def ensure(self) -> "StoragePaths":
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "database": str(self.database),
            "manifests": str(self.manifests),
            "corpus": str(self.corpus),
            "backups": str(self.backups),
            "logs": str(self.logs),
            "lock": str(self.lock),
        }


def _override(suffix: str, default: Path) -> Path:
    value = os.environ.get(f"{ENV_PREFIX}{suffix}")
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class PathCheck:
    """One path's verdict, with the reason when it failed."""

    name: str
    path: Path
    writable: bool
    reason: str = ""
    free_bytes: int | None = None


@dataclass(frozen=True)
class StorageValidation:
    checks: tuple[PathCheck, ...]
    free_bytes: int | None
    ok: bool

    def failures(self) -> tuple[PathCheck, ...]:
        return tuple(check for check in self.checks if not check.writable)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "free_bytes": self.free_bytes,
            "checks": [
                {
                    "name": check.name,
                    "path": str(check.path),
                    "writable": check.writable,
                    "reason": check.reason,
                }
                for check in self.checks
            ],
        }


def validate(paths: StoragePaths, *, require_free_bytes: int = 64 * 1024 * 1024) -> StorageValidation:
    """Prove each location is writable by writing to it.

    `exists()` and `os.access` both pass on a read-only mount, a full volume and
    a stale NFS handle. Writing a byte and removing it is the only check that
    corresponds to what the collector will actually do.
    """
    checks: list[PathCheck] = []
    for name, directory in (
        ("root", paths.root),
        ("manifests", paths.manifests),
        ("corpus", paths.corpus),
        ("backups", paths.backups),
        ("logs", paths.logs),
        ("database_parent", paths.database.parent),
        ("lock_parent", paths.lock.parent),
    ):
        checks.append(_probe(name, directory))

    free = _free_bytes(paths.root)
    if free is not None and free < require_free_bytes:
        checks.append(
            PathCheck(
                name="free_space",
                path=paths.root,
                writable=False,
                reason=f"{free:,} bytes free, below the {require_free_bytes:,} minimum",
                free_bytes=free,
            )
        )

    return StorageValidation(
        checks=tuple(checks),
        free_bytes=free,
        ok=all(check.writable for check in checks),
    )


def _probe(name: str, directory: Path) -> PathCheck:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".btc-intel-write-probe"
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return PathCheck(name=name, path=directory, writable=False, reason=f"{type(error).__name__}: {error}")
    return PathCheck(name=name, path=directory, writable=True)


def _free_bytes(path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(path).free)
    except OSError:  # pragma: no cover - unusual filesystems
        return None


def require_usable(paths: StoragePaths, *, require_free_bytes: int = 64 * 1024 * 1024) -> StorageValidation:
    """Validate, and refuse to continue if any location is unusable."""
    result = validate(paths, require_free_bytes=require_free_bytes)
    if not result.ok:
        detail = "; ".join(f"{check.name} at {check.path}: {check.reason}" for check in result.failures())
        raise ConfigurationError(
            f"persistent storage is not usable, refusing to start: {detail}. "
            "A collector that cannot persist would run cleanly and accumulate nothing."
        )
    return result


def looks_ephemeral(path: Path) -> str | None:
    """Warn when state is configured under a directory that will not survive.

    A heuristic, and it says so. It exists because the single most expensive
    misconfiguration here produces no error at all -- months of collection into
    a container layer that is discarded on the next deploy.
    """
    resolved = str(path.resolve()).replace("\\", "/").casefold()
    for marker, why in (
        ("/tmp/", "/tmp is cleared on reboot on most systems"),
        ("/var/tmp/", "/var/tmp is not guaranteed to survive"),
        ("/dev/shm", "/dev/shm is memory-backed and lost on restart"),
        ("/appdata/local/temp", "the Windows temp directory is cleared periodically"),
    ):
        if marker in resolved:
            return f"{path} looks ephemeral: {why}"
    return None


__all__ = [
    "DEFAULT_ROOT",
    "ENV_PREFIX",
    "PATH_ENV_SUFFIXES",
    "PATH_ENV_VARIABLES",
    "PathCheck",
    "StoragePaths",
    "StorageValidation",
    "looks_ephemeral",
    "require_usable",
    "validate",
]
