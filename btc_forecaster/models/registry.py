"""Model registry: build models by name, and report what is unavailable.

Registration is by *factory*, not by instance, for two reasons: models are
stateful once fitted, and the heavy ones must not be imported until something
actually asks for them. Asking the registry for a model whose dependency is
missing produces a clear message naming the extra to install, rather than an
ImportError from three layers down.

:func:`available` lets a run record which models could and could not be built,
so a comparison table that is missing a row says why.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import ForecastModel, MissingDependencyError

ModelFactory = Callable[..., ForecastModel]


@dataclass(frozen=True)
class Registration:
    name: str
    factory: ModelFactory
    family: str
    requires: tuple[str, ...]
    description: str

    def is_available(self) -> tuple[bool, str | None]:
        import importlib.util

        for module in self.requires:
            if importlib.util.find_spec(module) is None:
                return False, module
        return True, None


_REGISTRY: dict[str, Registration] = {}


def register(
    name: str,
    factory: ModelFactory,
    *,
    family: str,
    requires: tuple[str, ...] = (),
    description: str = "",
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"model {name!r} is already registered")
    _REGISTRY[name] = Registration(
        name=name, factory=factory, family=family, requires=requires, description=description
    )


def build(name: str, /, **kwargs) -> ForecastModel:
    """Construct a registered model, or explain why it cannot be built.

    The registry key becomes the model's canonical identity: unless the caller
    passes an explicit ``name``, the built model is renamed to the key it was
    requested under. Without this, a model whose class picks its own name from
    its hyperparameters (ArimaModel defaults to ``"arima(1, 1, 1)"``) would not
    match the registry key used to request it, and any lookup by configured name
    -- ``primary_model``, ``baseline_model``, a skill-table row -- would silently
    miss. Hyperparameters remain visible in ``describe()`` and in the forecast
    metadata, so nothing is lost by fixing the identity.

    The registry key is positional-only so that ``name`` in ``**kwargs`` reaches
    the model constructor rather than colliding with this parameter.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")

    registration = _REGISTRY[name]
    ok, missing = registration.is_available()
    if not ok:
        extra = _EXTRA_FOR_MODULE.get(missing or "", "all")
        raise MissingDependencyError(
            f"model {name!r} requires {missing!r}, which is not installed.\n"
            f'    pip install "btc-forecaster[{extra}]"'
        )

    model = registration.factory(**kwargs)
    if "name" not in kwargs:
        model._name = name
    return model


def names(family: str | None = None) -> list[str]:
    return sorted(n for n, r in _REGISTRY.items() if family is None or r.family == family)


def families() -> list[str]:
    return sorted({r.family for r in _REGISTRY.values()})


def available() -> dict[str, dict]:
    """Every registered model with its availability, for the run manifest."""
    report = {}
    for name, registration in sorted(_REGISTRY.items()):
        ok, missing = registration.is_available()
        report[name] = {
            "family": registration.family,
            "available": ok,
            "missing_dependency": missing,
            "requires": list(registration.requires),
            "description": registration.description,
        }
    return report


def build_available(model_names: list[str], **kwargs) -> tuple[list[ForecastModel], dict[str, str]]:
    """Build what can be built; report the rest instead of raising.

    A comparison run should still produce a table when an optional model is
    missing -- with that model's absence recorded, not silently dropped.
    """
    built: list[ForecastModel] = []
    skipped: dict[str, str] = {}
    for name in model_names:
        try:
            built.append(build(name, **kwargs))
        except MissingDependencyError as exc:
            skipped[name] = str(exc).splitlines()[0]
    return built, skipped


_EXTRA_FOR_MODULE = {
    "prophet": "models",
    "xgboost": "models",
    "sklearn": "models",
    "arch": "models",
    "statsmodels": "",
}


def _register_defaults() -> None:
    from .baselines import HistoricalMeanReturn, RandomWalk, RandomWalkWithDrift

    register(
        "random_walk",
        RandomWalk,
        family="baseline",
        description="P[T+h] = P[T]. The hypothesis every other model must disprove.",
    )
    register(
        "random_walk_drift",
        RandomWalkWithDrift,
        family="baseline",
        description="Random walk with classical endpoint drift.",
    )
    register(
        "historical_mean_return",
        HistoricalMeanReturn,
        family="baseline",
        description="Compounds the trailing arithmetic mean simple return.",
    )

    def _arima(**kwargs):
        from .statistical import ArimaModel

        return ArimaModel(**kwargs)

    def _arima_auto(**kwargs):
        from .statistical import ArimaModel

        kwargs.setdefault("order", None)
        return ArimaModel(**kwargs)

    def _sarimax(**kwargs):
        from .statistical import SarimaxModel

        return SarimaxModel(**kwargs)

    def _ets(**kwargs):
        from .statistical import EtsModel

        return EtsModel(**kwargs)

    register(
        "arima",
        _arima,
        family="statistical",
        requires=("statsmodels",),
        description="ARIMA(1,1,1) on log price with analytic intervals.",
    )
    register(
        "arima_auto",
        _arima_auto,
        family="statistical",
        requires=("statsmodels",),
        description="ARIMA with order chosen by AIC inside each fold.",
    )
    register(
        "sarimax",
        _sarimax,
        family="statistical",
        requires=("statsmodels",),
        description="Seasonal ARIMA with exogenous dynamic regression seam.",
    )
    register(
        "ets",
        _ets,
        family="statistical",
        requires=("statsmodels",),
        description="Damped additive-trend exponential smoothing on log price.",
    )

    def _prophet(**kwargs):
        from .prophet_model import ProphetModel

        return ProphetModel(**kwargs)

    def _hybrid(**kwargs):
        from .hybrid import ProphetXgboostHybrid

        return ProphetXgboostHybrid(**kwargs)

    register(
        "prophet",
        _prophet,
        family="structural",
        requires=("prophet",),
        description="Prophet trend + seasonality on log price (preserved from the original pipeline).",
    )
    register(
        "prophet_xgb_hybrid",
        _hybrid,
        family="hybrid",
        requires=("prophet", "xgboost"),
        description="Prophet baseline with XGBoost residual correction (the original headline model).",
    )

    def _xgb_causal(**kwargs):
        from .challengers import XgboostCausalRetuned

        return XgboostCausalRetuned(**kwargs)

    register(
        "xgboost_causal_retuned",
        _xgb_causal,
        family="challenger",
        requires=("xgboost",),
        description=(
            "XGBOOST_CAUSAL_RETUNED: direct XGBoost on next-bar log return, "
            "hyperparameters selected by nested inner validation per fold."
        ),
    )


_register_defaults()

__all__ = [
    "ModelFactory",
    "Registration",
    "available",
    "build",
    "build_available",
    "families",
    "names",
    "register",
]
