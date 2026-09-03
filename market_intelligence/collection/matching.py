"""How a planned query's terms are matched against a feed entry's text.

Split out of `syndication` because it is a relevance decision, not a transport
one, and because getting it wrong is quiet: the documents a loose rule admits
are real releases from a real publisher, so nothing downstream looks broken.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Bumped when candidate matching changes what it admits. A corpus collected
#: under one rule and extended under another has a discontinuity in it that no
#: downstream count can see, so the version is recorded rather than inferred.
MATCHING_CONTRACT_VERSION = "matching-v1-word-start"


def term_pattern(terms: Sequence[str]) -> re.Pattern[str] | None:
    """Match terms at a word start, allowing suffixes. `None` when unfiltered.

    Plain substring containment admitted documents nobody asked for. The
    planner emits `"US Treasury" sanctions`, whose terms include the two-letter
    `us`, and containment then matched "SEC Anno(us)nces", "B(us)iness",
    "Foc(us)" -- 18 of 382 candidate matches in one measured live cycle, 4.7%,
    every one an SEC roundtable or advisory committee unrelated to the query.
    That is precision loss, and it is invisible, because the documents it
    admits are genuine SEC releases.

    Anchoring at a word *start* rather than requiring a whole word is
    deliberate. A whole-word rule would stop matching "regulations" for the
    term `regulation` and "bitcoin's" for `bitcoin`, throwing away recall this
    filter should keep. A start-anchored prefix keeps both and still refuses to
    match inside an unrelated word.

    Nothing here widens what is admitted. Every entry this rejects was already
    being admitted only by accident.
    """
    if not terms:
        return None
    alternation = "|".join(re.escape(term) for term in terms)
    return re.compile(r"\b(?:" + alternation + ")")


__all__ = ["MATCHING_CONTRACT_VERSION", "term_pattern"]
