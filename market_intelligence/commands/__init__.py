"""The commands `btc-intel` can run, and what running one costs.

`_main` was a 230-line chain of `if args.command == ...` covering 27 commands,
which made three things hard that a registry makes easy: seeing which commands
exist, seeing which of them touch the corpus, and adding one without reading
the other twenty-six.

The split here is by what a command *does to the corpus*, because that is what
determines its blast radius:

* `collection` -- writes new evidence into it
* `corpus` -- verifies, snapshots, backs up and restores it
* `corrections` -- appends to its correction ledger
* `datasets` -- projects it into replay datasets and features
* `ops` -- reports on the machine running it, and writes nothing
* `demos` -- runs entirely on fixtures and touches no real corpus

Two registries rather than one, and the distinction is load-bearing. Opening a
store creates the database file if it is absent, so a command that runs against
fixtures or manages its own storage must never be handed one: `btc-intel demo`
opening `./btc-intelligence.duckdb` would leave a stray empty corpus in
whatever directory it was run from.

Exit codes are a documented interface -- 0 ran, 2 failed, 3 another collector
holds the lock, 4 nothing was due -- so every handler returns one and none of
them exits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from ..services import IntelligenceReadService
from ..storage import IntelligenceStore


@dataclass(frozen=True)
class CommandContext:
    """What a command that reads the corpus is given.

    `reads` is the same service the API uses. Handing both to every handler
    means neither surface can quietly grow its own way of answering a question
    the other already answers.
    """

    args: argparse.Namespace
    store: IntelligenceStore
    reads: IntelligenceReadService


#: A command that needs the corpus.
StoreHandler = Callable[[CommandContext], int]

#: A command that must not be handed one -- see the note above about stray
#: database files.
StorelessHandler = Callable[[argparse.Namespace], int]


def _registry() -> tuple[dict[str, StorelessHandler], dict[str, StoreHandler]]:
    """Built inside a function so importing this package does not import every
    command's dependencies. `btc-intel --help` should not need duckdb."""
    from . import collection, corpus, corrections, datasets, demos, ops

    storeless: dict[str, StorelessHandler] = {
        "demo": demos.demo,
        "gold-report": demos.gold_report,
        "collect-scheduled": collection.collect_scheduled,
    }
    with_store: dict[str, StoreHandler] = {
        "collect": collection.collect,
        "extract": collection.extract,
        "aggregate": collection.aggregate,
        "backfill": collection.backfill,
        "corpus-status": corpus.corpus_status,
        "corpora": corpus.corpora,
        "corpus-verify": corpus.corpus_verify,
        "corpus-backup": corpus.corpus_backup,
        "corpus-restore": corpus.corpus_restore,
        "corpus-correct": corrections.apply_corrections,
        "corpus-corrections": corrections.report_corrections,
        "replay": datasets.replay,
        "dataset": datasets.dataset,
        "replay-dataset": datasets.replay_dataset,
        "catalog": datasets.catalog,
        "health": ops.health,
        # `ops-paths` and `providers` read neither, but they opened a store
        # before this split and opening one creates the file. Keeping them here
        # keeps that side effect exactly as it was; a refactor is not the place
        # to change what a command leaves behind on disk.
        "ops-paths": ops.ops_paths,
        "providers": ops.providers,
        "quality": ops.quality,
        "ops-status": ops.ops_status,
        "ops-watch": ops.ops_watch,
    }
    return storeless, with_store


def command_names() -> tuple[str, ...]:
    """Every command the dispatcher can run. The parser is checked against it."""
    storeless, with_store = _registry()
    return tuple(sorted({*storeless, *with_store}))


def dispatch(args: argparse.Namespace) -> int:
    """Run one command and return its exit code.

    The store is opened only for the commands that need it and is always
    closed, including when a handler raises -- a DuckDB file left open by a
    crashed process is a file the next run has to recover rather than read.
    """
    storeless, with_store = _registry()

    handler = storeless.get(args.command)
    if handler is not None:
        return handler(args)

    with_store_handler = with_store.get(args.command)
    if with_store_handler is None:
        # argparse rejects an unknown command before this, so reaching here
        # means a command was declared to the parser and never registered.
        return 1

    store = IntelligenceStore(args.db)
    try:
        return with_store_handler(CommandContext(args, store, IntelligenceReadService(store)))
    finally:
        store.close()


__all__ = [
    "CommandContext",
    "StoreHandler",
    "StorelessHandler",
    "command_names",
    "dispatch",
]
