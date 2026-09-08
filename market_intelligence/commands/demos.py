"""Commands that run entirely on fixtures.

Neither touches a real corpus, contacts a network or needs a credential, which
is why both are registered as commands that must not be handed a store: opening
one would create a database file in whatever directory they were run from, and
the whole point of these two is that they leave nothing behind except what they
were asked to write.

Both are exercised in CI on every push, so a fixture that stops matching the
code fails there rather than in a demonstration.
"""

from __future__ import annotations

import argparse
import json

from ..extractors import RuleBasedExtractor

#: The watchlist the gold-standard evaluation scores against. Aliases are given
#: only where the source text genuinely uses them.
GOLD_WATCHLIST = {"SEC": (), "Jerome Powell": ("Powell",), "CFTC": (), "Elon Musk": ()}


def demo(args: argparse.Namespace) -> int:
    from ..demo import run_offline_demo

    print(json.dumps(run_offline_demo(args.output_dir), indent=2))
    return 0


def gold_report(args: argparse.Namespace) -> int:
    from ..gold import write_gold_evaluation

    report = write_gold_evaluation(RuleBasedExtractor(GOLD_WATCHLIST), args.output)
    print(report.model_dump_json(indent=2))
    return 0


__all__ = ["GOLD_WATCHLIST", "demo", "gold_report"]
