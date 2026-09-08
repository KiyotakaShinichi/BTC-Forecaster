"""Persistence for the point-in-time intelligence corpus.

`market_intelligence.storage` used to be one 887-line module holding three
responsibilities that fail in different ways and are read for different reasons:
the DDL, every write, and every read. It is now a package, and the public
surface is unchanged -- `from market_intelligence.storage import
IntelligenceStore` resolves exactly as before, and so does every other symbol
the module exported.

* `schema.py` -- what a row is. The only DDL in the repository, idempotent,
  forward-only, non-destructive.
* `store.py` -- `IntelligenceStore`: the connection, its lifecycle, and every
  write. The guarantee is identity: first write wins, availability never moves,
  corrections append.
* `queries.py` -- every read, as functions over a connection. The guarantee is
  determinism: the same question returns the same answer in the same total
  order on every machine.
"""

from __future__ import annotations

from . import queries, schema
from .schema import (
    SCHEMA_DDL,
    SCHEMA_VERSION,
    SNAPSHOT_INSERT_CHUNK,
    is_ready,
    migrate,
    schema_version,
)
from .store import IntelligenceStore

__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "SNAPSHOT_INSERT_CHUNK",
    "IntelligenceStore",
    "is_ready",
    "migrate",
    "queries",
    "schema",
    "schema_version",
]
