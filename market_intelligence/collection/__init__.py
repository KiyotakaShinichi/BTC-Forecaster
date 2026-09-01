"""Track B4.1 — forward point-in-time intelligence collection.

B4 ended on HOLD with an empty corpus: the architecture was there, but no
provider could lawfully produce evidence, and nothing had ever been collected.
This package closes that gap. It does not try to prove predictive value — it
collects, validates, stores, monitors and exports evidence whose historical
availability is defensible, so that a future B4 run has something to test.

The rule everything here is built around:

    a document discovered today was available *today*, at retrieval, unless a
    provider supplies independently defensible evidence of an earlier moment.

`published_at` is not `available_at`. A press release dated three weeks ago that
this system first saw an hour ago became usable an hour ago, and treating its
publication date as its availability would fabricate three weeks of hindsight
into every study built on it.
"""

from __future__ import annotations

B41_COLLECTION_VERSION = "b41-collection-v1"

__all__ = ["B41_COLLECTION_VERSION"]
