"""Adapters: each module wraps one family behind the single zoo contract.

Every module here registers its models at import time, and
:mod:`btc_forecaster.research.registry` imports them exactly once. Nothing in
this package is imported eagerly by the zoo's own ``__init__``, so listing the
registry does not drag scikit-learn, statsmodels or the deep engine into a
process that only wanted to print a table.
"""

from __future__ import annotations

__all__: list[str] = []
