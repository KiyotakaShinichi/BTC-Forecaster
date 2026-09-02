"""B4.1-Ops — running the forward collector continuously, safely, for months.

No new research features. B4.1's scientific semantics are frozen: event
definitions, `available_at`, and B4's adequacy thresholds are untouched here.

What this package adds is the difference between a collector that works when you
run it and one you can leave running: a single-instance lock, an explicit
persistent-storage contract, crash recovery, backup and restore, a watchdog that
can tell a quiet day from a dead provider, and an integrity check that fails
closed.
"""

from __future__ import annotations

OPS_VERSION = "b41-ops-v1"

__all__ = ["OPS_VERSION"]
