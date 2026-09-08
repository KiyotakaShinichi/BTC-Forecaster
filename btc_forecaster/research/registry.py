"""The single source of truth for what is in the zoo.

Every list of models in this repository is generated from here: the runner, the
CLI's ``--list``, the capability matrix, the model cards, the documentation
table and the tests. A second hand-maintained list somewhere else is how a zoo
acquires a model that is documented and never run, or run and never documented.
A test asserts no such list exists.

Registration is by **factory**, never by instance, for two reasons: a fitted
model is stateful, and importing forty adapters -- several of which reach for
scikit-learn or statsmodels -- must not happen because somebody typed
``--help``.

A registration also carries its own **status**, and the vocabulary is
deliberately blunt:

``ACTIVE``                          runs in the default benchmark.
``OPTIONAL``                        runs only when its dependency is present.
``UNSUITABLE_FOR_CONSTRAINED_LAB``  cannot honestly run at n=1,000, with the
                                    reason recorded. Still registered, still
                                    listed, never silently dropped.

A model that fails at runtime is recorded as ``FAILED`` with its exception, and
one whose dependency is missing as ``SKIPPED_DEPENDENCY``. The one outcome the
registry makes impossible is a model that quietly is not there.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .contracts import Family, ModelStatus, ResourceClass, ZooModel

ZooFactory = Callable[[], ZooModel]


@dataclass(frozen=True)
class ZooRegistration:
    """One model's declaration, independent of whether it can run here."""

    model_id: str
    factory: ZooFactory
    family: Family
    resource_class: ResourceClass
    description: str
    requires: tuple[str, ...] = ()
    status: ModelStatus = ModelStatus.ACTIVE
    #: Required when status is UNSUITABLE_FOR_CONSTRAINED_LAB. A model excluded
    #: without a stated reason is indistinguishable from one that was forgotten.
    unsuitable_reason: str | None = None
    #: Free-form caveats that belong on the model card.
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status is ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB and not self.unsuitable_reason:
            raise ValueError(
                f"{self.model_id} is marked unsuitable without a reason; "
                "an unexplained exclusion is not a scientific statement"
            )

    def missing_dependency(self) -> str | None:
        for module in self.requires:
            if importlib.util.find_spec(module) is None:
                return module
        return None

    def is_available(self) -> bool:
        return self.missing_dependency() is None

    def effective_status(self) -> ModelStatus:
        """Status after checking what is actually installed."""
        if self.status is ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB:
            return self.status
        missing = self.missing_dependency()
        if missing is not None:
            return ModelStatus.SKIPPED_DEPENDENCY
        return self.status

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "family": self.family.value,
            "resource_class": self.resource_class.value,
            "description": self.description,
            "requires": list(self.requires),
            "declared_status": self.status.value,
            "effective_status": self.effective_status().value,
            "missing_dependency": self.missing_dependency(),
            "unsuitable_reason": self.unsuitable_reason,
            "notes": list(self.notes),
        }


#: Every adapter module, imported once on first registry use. This tuple is the
#: only place the set is written down -- a module absent from it is a module
#: whose models silently do not exist, which is precisely the failure the
#: registry is here to prevent.
ADAPTER_MODULES: tuple[str, ...] = ("baselines", "statistical", "volatility")

_REGISTRY: dict[str, ZooRegistration] = {}


def register(registration: ZooRegistration) -> ZooRegistration:
    if registration.model_id in _REGISTRY:
        raise ValueError(f"model {registration.model_id!r} is already registered")
    _REGISTRY[registration.model_id] = registration
    return registration


def get(model_id: str) -> ZooRegistration:
    _ensure_loaded()
    if model_id not in _REGISTRY:
        raise KeyError(f"unknown model {model_id!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[model_id]


def all_registrations() -> list[ZooRegistration]:
    _ensure_loaded()
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def model_ids(
    *,
    family: Family | None = None,
    statuses: Iterable[ModelStatus] | None = None,
) -> list[str]:
    wanted = set(statuses) if statuses is not None else None
    return [
        r.model_id
        for r in all_registrations()
        if (family is None or r.family is family)
        and (wanted is None or r.effective_status() in wanted)
    ]


def runnable_ids() -> list[str]:
    """What the default benchmark will actually attempt."""
    return model_ids(statuses={ModelStatus.ACTIVE, ModelStatus.OPTIONAL})


def families() -> list[Family]:
    return sorted({r.family for r in all_registrations()}, key=lambda f: f.value)


def build(model_id: str) -> ZooModel:
    registration = get(model_id)
    missing = registration.missing_dependency()
    if missing is not None:
        raise ImportError(
            f"model {model_id!r} requires {missing!r}, which is not installed"
        )
    model = registration.factory()
    if model.model_id != model_id:
        raise ValueError(
            f"registration key {model_id!r} disagrees with the model's own id "
            f"{model.model_id!r}; identity must be stated once"
        )
    return model


def capability_matrix() -> list[dict]:
    """Every model against every capability, built without fitting anything.

    Constructing a model is cheap; fitting is not. This gives the report its
    capability table and gives the tests something to assert against without a
    benchmark run.
    """
    from .contracts import Capability

    rows: list[dict] = []
    for registration in all_registrations():
        row: dict = {
            "model_id": registration.model_id,
            "family": registration.family.value,
            "status": registration.effective_status().value,
        }
        if registration.is_available() and registration.status is not (
            ModelStatus.UNSUITABLE_FOR_CONSTRAINED_LAB
        ):
            model = registration.factory()
            row["preprocessing"] = model.preprocessing.value
            for capability in Capability:
                row[capability.value] = capability in model.capabilities
        else:
            row["preprocessing"] = None
            for capability in Capability:
                row[capability.value] = None
        rows.append(row)
    return rows


def summary() -> dict:
    """Counts by status and family, for the manifest and the final report."""
    registrations = all_registrations()
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for registration in registrations:
        status = registration.effective_status().value
        by_status[status] = by_status.get(status, 0) + 1
        family = registration.family.value
        by_family[family] = by_family.get(family, 0) + 1
    return {
        "total_registered": len(registrations),
        "by_status": dict(sorted(by_status.items())),
        "by_family": dict(sorted(by_family.items())),
        "runnable": len(runnable_ids()),
    }


_LOADED = False


def _ensure_loaded() -> None:
    """Import the adapter modules once, on first use.

    Each adapter module registers its own models at import. Doing it lazily
    keeps ``--help`` free of scikit-learn, and doing it exactly once keeps the
    duplicate-registration guard meaningful.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import adapters

    for module in ADAPTER_MODULES:
        importlib.import_module(f"{adapters.__name__}.{module}")


__all__ = [
    "ZooFactory",
    "ZooRegistration",
    "all_registrations",
    "build",
    "capability_matrix",
    "families",
    "get",
    "model_ids",
    "register",
    "runnable_ids",
    "summary",
]
