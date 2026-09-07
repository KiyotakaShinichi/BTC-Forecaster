"""The ForecastModel contract.

Every model in the platform -- naive baseline, ARIMA, ETS, Prophet, the XGBoost
hybrid, and anything added later -- implements this interface, so the
walk-forward engine can score all of them through identical folds.

The contract is deliberately shaped to make leakage structurally difficult:

* :meth:`ForecastModel.fit` receives a :class:`TrainingWindow` and nothing else.
  A model cannot look at data past its training window because it was never
  handed any. Feature selection, hyperparameter choice and preprocessing all
  happen inside ``fit``, which means the walk-forward engine refits them per
  fold for free.
* The forecast origin is *derived* from the training window's last bar rather
  than passed separately, so an origin can never disagree with the data behind
  it.
* :meth:`ForecastModel.predict` takes only a horizon. Future exogenous inputs
  must be explicitly declared and are restricted to regressors that are known in
  advance (see ``future_exog``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.contracts import validate_market_frame
from ..features.spec import log_price, log_returns, simple_returns
from ..timebase import ForecastOrigin, HorizonSpec, to_utc_index


class NotFittedError(RuntimeError):
    """Raised when a model is asked to predict before it has been fitted."""


class MissingDependencyError(ImportError):
    """Raised when a model's optional third-party dependency is not installed."""


@dataclass(frozen=True)
class TrainingWindow:
    """Everything a model is allowed to see, and nothing else.

    ``exog`` is the seam for external signals (Track B). Any column supplied
    here must already be point-in-time correct: value at bar D knowable at D's
    close. The platform does not fix up lags for you -- the producer declares
    them via ``FeatureSpec.publication_lag``.
    """

    frame: pd.DataFrame
    exog: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        validate_market_frame(self.frame)
        if self.exog is not None:
            exog = self.exog
            if not exog.index.equals(self.frame.index):
                raise ValueError("exog must be indexed identically to the market frame")

    @property
    def origin(self) -> ForecastOrigin:
        """The forecast origin implied by this window: its last observed bar."""
        return ForecastOrigin(self.frame.index.max())

    @property
    def close(self) -> pd.Series:
        return self.frame["close"]

    @property
    def log_close(self) -> pd.Series:
        return log_price(self.frame)

    @property
    def returns(self) -> pd.Series:
        return simple_returns(self.frame)

    @property
    def log_returns(self) -> pd.Series:
        return log_returns(self.frame)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    def __len__(self) -> int:
        return len(self.frame)

    def tail(self, bars: int) -> TrainingWindow:
        return TrainingWindow(
            frame=self.frame.iloc[-bars:],
            exog=None if self.exog is None else self.exog.iloc[-bars:],
        )


@dataclass(frozen=True)
class ForecastResult:
    """A forecast, in price space, over a known set of target bars."""

    model: str
    origin: ForecastOrigin
    point: pd.Series
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    interval_level: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.point.index.min() <= self.origin.last_observed_bar:
            raise ValueError(
                f"{self.model} forecast includes bar {self.point.index.min()}, "
                f"which is not after its origin {self.origin.last_observed_bar}"
            )
        for name, band in (("lower", self.lower), ("upper", self.upper)):
            if band is not None and not band.index.equals(self.point.index):
                raise ValueError(f"{name} band must share the point forecast's index")
        if self.has_interval:
            if bool((self.upper < self.lower).any()):  # type: ignore[operator]
                raise ValueError(f"{self.model} produced an inverted prediction interval")

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.point.index)

    @property
    def horizon(self) -> int:
        return len(self.point)

    @property
    def has_interval(self) -> bool:
        return self.lower is not None and self.upper is not None

    def to_frame(self) -> pd.DataFrame:
        data = {"point": self.point}
        if self.has_interval:
            data["lower"] = self.lower
            data["upper"] = self.upper
        out = pd.DataFrame(data)
        out.index.name = "date"
        return out

    def head(self, n: int) -> ForecastResult:
        """The first ``n`` steps. Used to score a long forecast at a short horizon."""
        return ForecastResult(
            model=self.model,
            origin=self.origin,
            point=self.point.iloc[:n],
            lower=None if self.lower is None else self.lower.iloc[:n],
            upper=None if self.upper is None else self.upper.iloc[:n],
            interval_level=self.interval_level,
            metadata=self.metadata,
        )


class ForecastModel(ABC):
    """Base class for every forecasting model.

    Subclasses implement :meth:`_fit` and :meth:`_predict`. The public
    :meth:`fit` / :meth:`predict` handle the bookkeeping that must not vary
    between models: recording the origin, refusing to predict before fitting,
    and stamping results with the model's name.
    """

    #: Optional third-party imports this model needs, for a clear error message.
    requires: tuple[str, ...] = ()

    def __init__(self, name: str | None = None) -> None:
        self._name = name or self.__class__.__name__
        self._window: TrainingWindow | None = None
        self._fitted = False

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_probabilistic(self) -> bool:
        """Whether :meth:`predict` returns prediction intervals."""
        return True

    def describe(self) -> dict:
        """Hyperparameters worth recording in a run manifest."""
        return {"name": self.name, "type": self.__class__.__name__}

    # -- lifecycle --------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def window(self) -> TrainingWindow:
        if self._window is None:
            raise NotFittedError(f"{self.name} has not been fitted")
        return self._window

    @property
    def origin(self) -> ForecastOrigin:
        """The forecast origin implied by the training data."""
        return self.window.origin

    def fit(self, window: TrainingWindow) -> ForecastModel:
        if len(window) < self.min_train_bars:
            raise ValueError(
                f"{self.name} needs at least {self.min_train_bars} bars, got {len(window)}"
            )
        self._window = window
        self._fit(window)
        self._fitted = True
        return self

    def predict(
        self,
        horizon: HorizonSpec | int,
        *,
        future_exog: pd.DataFrame | None = None,
    ) -> ForecastResult:
        """Forecast ``horizon`` bars ahead of the training window's last bar.

        ``future_exog`` may only carry regressors that are genuinely known in
        advance -- calendar effects, scheduled protocol events, announced policy
        dates. Anything that must itself be forecast does not belong here; using
        a realised future value would be leakage of the most direct kind.
        """
        if not self._fitted:
            raise NotFittedError(f"{self.name} has not been fitted")

        spec = horizon if isinstance(horizon, HorizonSpec) else HorizonSpec(int(horizon))
        target_bars = spec.bars_from(self.origin)

        if future_exog is not None:
            supplied = to_utc_index(future_exog.index)
            if not supplied.equals(target_bars):
                raise ValueError("future_exog must be indexed by exactly the target bars")

        result = self._predict(spec, target_bars, future_exog)

        if not result.index.equals(target_bars):
            raise ValueError(
                f"{self.name} returned {len(result.index)} bars that do not match the "
                f"{len(target_bars)} requested target bars"
            )
        return result

    def fit_predict(
        self,
        window: TrainingWindow,
        horizon: HorizonSpec | int,
        *,
        future_exog: pd.DataFrame | None = None,
    ) -> ForecastResult:
        return self.fit(window).predict(horizon, future_exog=future_exog)

    # -- to implement -----------------------------------------------------

    @property
    def min_train_bars(self) -> int:
        """Minimum training bars this model can be fitted on."""
        return 2

    @abstractmethod
    def _fit(self, window: TrainingWindow) -> None: ...

    @abstractmethod
    def _predict(
        self,
        horizon: HorizonSpec,
        target_bars: pd.DatetimeIndex,
        future_exog: pd.DataFrame | None,
    ) -> ForecastResult: ...

    # -- helpers for subclasses -------------------------------------------

    def _require(self, module: str, extra: str):
        try:
            return __import__(module)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingDependencyError(
                f"{self.name} requires {module!r}. Install the optional extra:\n"
                f'    pip install "btc-forecaster[{extra}]"'
            ) from exc

    def _result(
        self,
        point: np.ndarray | pd.Series,
        target_bars: pd.DatetimeIndex,
        *,
        lower: np.ndarray | pd.Series | None = None,
        upper: np.ndarray | pd.Series | None = None,
        interval_level: float | None = None,
        metadata: dict | None = None,
    ) -> ForecastResult:
        def as_series(values) -> pd.Series | None:
            if values is None:
                return None
            series = pd.Series(np.asarray(values, dtype=float), index=target_bars)
            series.index.name = "date"
            return series

        return ForecastResult(
            model=self.name,
            origin=self.origin,
            point=as_series(point),  # type: ignore[arg-type]
            lower=as_series(lower),
            upper=as_series(upper),
            interval_level=interval_level,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "fitted" if self._fitted else "unfitted"
        return f"<{self.__class__.__name__} {self.name!r} ({state})>"


def random_walk_bands(
    last_log_price: float,
    drift_per_step: float,
    sigma_per_step: float,
    steps: int,
    *,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point forecast and interval for a log-price random walk with drift.

    Under ``log P_{T+h} = log P_T + h*mu + sum of h iid N(0, sigma^2)`` shocks,
    the h-step variance is ``h * sigma^2``, so the band widens with ``sqrt(h)``.

    This is the property the pre-Track-A Monte Carlo was missing: it added a
    *single* GARCH-scaled shock at each horizon rather than accumulating shocks
    along the path, producing intervals that stayed roughly constant width
    hundreds of days out.
    """
    from scipy.stats import norm

    if steps < 1:
        raise ValueError("steps must be >= 1")

    h = np.arange(1, steps + 1, dtype=float)
    mean_log = last_log_price + drift_per_step * h
    sd_log = sigma_per_step * np.sqrt(h)
    z = float(norm.ppf(0.5 + level / 2.0))

    point = np.exp(mean_log)
    lower = np.exp(mean_log - z * sd_log)
    upper = np.exp(mean_log + z * sd_log)
    return point, lower, upper


__all__ = [
    "ForecastModel",
    "ForecastResult",
    "MissingDependencyError",
    "NotFittedError",
    "TrainingWindow",
    "random_walk_bands",
]
