"""Commands that project the corpus into replay datasets and feature matrices.

None of these write to the corpus. What they produce is a derived artefact plus
a manifest recording the file hash and the row count, and the manifest is the
point: a dataset without one is a file, not evidence.

`replay-dataset` uses the same `HistoricalDatasetService` the API calls, so the
two cannot drift into building different matrices from the same origins.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..catalog import DatasetCatalog
from ..errors import ConfigurationError
from ..historical import HistoricalDatasetService
from ..origins import OriginFrequency, generate_origins, parse_origin
from ..replay_dataset import ReplayDatasetBuilder, ReplayMode
from ..services import SnapshotService
from . import CommandContext


def replay(ctx: CommandContext) -> int:
    snapshot = SnapshotService(ctx.store).build_snapshot(
        ctx.args.origin, {}, ctx.args.config_fingerprint
    )
    print(snapshot.model_dump_json(indent=2))
    return 0


def dataset(ctx: CommandContext) -> int:
    args = ctx.args
    source = Path(args.origins)
    try:
        origins = [
            parse_origin(value.strip())
            for value in source.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    except ValueError as error:
        # An unreadable origins file is an operational failure, so it exits 2
        # with a sentence. It used to escape as a traceback and exit 1, which
        # the documented exit-code contract does not mention.
        raise ConfigurationError(f"origins file {source}: {error}") from error
    manifest = ReplayDatasetBuilder(ctx.store).build(
        origins,
        args.output,
        args.manifest,
        {},
        args.config_fingerprint,
        args.format,
        mode=ReplayMode(args.mode),
    )
    print(manifest.model_dump_json(indent=2))
    return 0


def replay_dataset(ctx: CommandContext) -> int:
    args = ctx.args
    # Same service the API calls, so the two cannot drift.
    service = (
        HistoricalDatasetService(ctx.store, chunk_size=args.chunk_size)
        if args.chunk_size
        else HistoricalDatasetService(ctx.store)
    )
    origins = generate_origins(args.start, args.end, OriginFrequency(args.frequency))
    if args.extend:
        result = service.extend(
            args.extend,
            origins,
            args.output,
            args.manifest,
            {},
            args.config_fingerprint,
            export_format=args.format,
            mode=ReplayMode(args.mode),
        )
    else:
        result = service.build(
            origins,
            args.output,
            args.manifest,
            {},
            args.config_fingerprint,
            export_format=args.format,
            mode=ReplayMode(args.mode),
            resume=not args.no_resume,
        )
    print(
        json.dumps(
            {
                "dataset_id": result.manifest.dataset_id,
                "rows": result.manifest.row_count,
                "rows_appended": result.rows_appended,
                "chunks": result.chunk_count,
                "resumed_chunks": result.resumed_chunks,
                "mode": result.manifest.mode,
                "origin_frequency": result.catalog_entry.origin_frequency,
                "output": str(result.output_path),
                "manifest": str(result.manifest_path),
                "file_hash": result.manifest.file_hash,
            },
            indent=2,
        )
    )
    return 0


def catalog(ctx: CommandContext) -> int:
    entries = DatasetCatalog(ctx.store.connection).list_datasets(limit=ctx.args.limit)
    print(json.dumps([entry.model_dump(mode="json") for entry in entries], indent=2))
    return 0


__all__ = ["catalog", "dataset", "replay", "replay_dataset"]
