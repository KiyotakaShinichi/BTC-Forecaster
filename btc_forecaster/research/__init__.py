"""Track A6 — the resource-constrained model zoo.

This package is **exploratory research infrastructure**. It exists to answer a
breadth question the platform could not previously answer: given a fixed, small
training budget, how do forty forecasting models from six families actually
behave on this series, and do they make *different* mistakes?

It is not a promotion study, and nothing here can promote anything.

Three sentences carry the whole scientific status of this package:

1. A6 fits every model on **1,000 deterministically chosen training rows** and
   scores them on a frozen evaluation block. That is a sample-size regime in
   which almost nothing is decidable.
2. **A2 remains the authoritative historical evidence** — 36 walk-forward folds,
   8 models, 0 promotions, nothing beating the random walk. A 1,000-row result
   does not supersede it and is not comparable to it.
3. Every A6 model is `EXPLORATORY`. The paper-trading engine reads A2's
   evidence, finds no promoted model, and stays fail-closed. Nothing in this
   package changes that, and a test asserts it.

The reason to build it anyway is that breadth is cheap and ignorance is not:
knowing that a GRU and a Ridge make correlated errors on this series, or that
the whole zoo is indistinguishable from a random walk at n=1,000, is worth
knowing before anyone proposes a preregistered study.
"""

from __future__ import annotations

__all__: list[str] = []
