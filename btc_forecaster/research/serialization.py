"""Save a fitted model, reload it, and prove it gives the same answer.

A benchmark that cannot be reloaded is a benchmark whose results expire with the
process that produced them. So every model in the zoo declares
``Capability.SERIALIZE`` and this module makes the claim checkable: fit,
serialize, reload, predict, and assert the forecasts are **bit-identical**.

Bit-identical rather than close. A tolerance would hide the failure this is for:
a reloaded model that lost its scaler, its calibration block or its random state
does not produce slightly different numbers, it produces plausible ones. Every
family here round-trips exactly, so exactness is the honest bar.

Pickle, deliberately. The alternatives -- a hand-written serializer per family,
or ONNX -- would each be a second implementation of every model's state, and a
second thing that can disagree with the first. The trade-off is that a pickle is
only loadable by compatible code, so the artifact hash is recorded alongside the
package versions that wrote it.

**A pickle is executable.** These artifacts are research outputs of this
repository, written and read by it. Loading one from an untrusted source would
run whatever it contains, which is why `load_artifact` verifies the recorded
hash before unpickling and refuses on a mismatch.
"""

from __future__ import annotations

import hashlib
import pickle
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .contracts import Capability, EvaluationContext, ZooModel

#: Protocol 5 for out-of-band buffers on the larger numpy arrays.
PICKLE_PROTOCOL = 5


class ArtifactIntegrityError(RuntimeError):
    """A serialized model does not match the hash recorded for it."""


@dataclass(frozen=True)
class Artifact:
    """A serialized model and everything needed to trust it later."""

    model_id: str
    payload: bytes
    sha256: str
    bytes_written: int
    created_at: str
    environment: dict

    def as_dict(self) -> dict:
        """The manifest entry. Deliberately excludes the payload."""
        return {
            "model_id": self.model_id,
            "sha256": self.sha256,
            "bytes": self.bytes_written,
            "created_at": self.created_at,
            "environment": self.environment,
            "format": f"pickle protocol {PICKLE_PROTOCOL}",
            "note": (
                "loadable only by compatible code; the package versions that "
                "wrote it are recorded because a pickle is not a data format"
            ),
        }


def _environment() -> dict:
    import numpy as _np

    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": _np.__version__,
    }
    for name in ("sklearn", "statsmodels", "arch", "xgboost"):
        try:
            versions[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            versions[name] = "absent"
    return versions


def serialize(model: ZooModel) -> Artifact:
    """Serialize a fitted model and hash the exact bytes."""
    if not model.supports(Capability.SERIALIZE):
        raise NotImplementedError(f"{model.model_id} does not declare SERIALIZE")
    payload = pickle.dumps(model, protocol=PICKLE_PROTOCOL)
    return Artifact(
        model_id=model.model_id,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes_written=len(payload),
        created_at=datetime.now(UTC).isoformat(),
        environment=_environment(),
    )


def deserialize(artifact: Artifact) -> ZooModel:
    """Reload, verifying the hash first.

    The check is not paranoia about disk corruption. Unpickling executes, so the
    hash is the only thing standing between a swapped artifact and arbitrary
    code -- and verifying it costs a microsecond.
    """
    digest = hashlib.sha256(artifact.payload).hexdigest()
    if digest != artifact.sha256:
        raise ArtifactIntegrityError(
            f"{artifact.model_id}: recorded {artifact.sha256[:16]}, "
            f"payload hashes to {digest[:16]}"
        )
    return pickle.loads(artifact.payload)  # noqa: S301 -- hash-verified, self-written


def write_artifact(model: ZooModel, directory: Path | str) -> Artifact:
    artifact = serialize(model)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{artifact.model_id}.pkl").write_bytes(artifact.payload)
    return artifact


def load_artifact(model_id: str, directory: Path | str, *, sha256: str) -> ZooModel:
    payload = (Path(directory) / f"{model_id}.pkl").read_bytes()
    return deserialize(
        Artifact(
            model_id=model_id,
            payload=payload,
            sha256=sha256,
            bytes_written=len(payload),
            created_at="",
            environment={},
        )
    )


def round_trip_is_exact(model: ZooModel, context: EvaluationContext) -> bool:
    """Fit -> serialize -> reload -> predict, compared bit for bit.

    Exactness rather than a tolerance, because the failure this catches -- a
    reloaded model that lost its scaler or its calibration -- does not produce
    slightly different numbers. It produces plausible ones.
    """
    before = model.predict(context)
    after = deserialize(serialize(model)).predict(context)
    if not np.array_equal(before.point, after.point):
        return False
    if (before.quantiles is None) != (after.quantiles is None):
        return False
    if before.quantiles is not None and after.quantiles is not None:
        for level, values in before.quantiles.items():
            if not np.array_equal(values, after.quantiles[level]):
                return False
    for a, b in (
        (before.direction_probability, after.direction_probability),
        (before.variance, after.variance),
    ):
        if (a is None) != (b is None):
            return False
        if a is not None and b is not None and not np.array_equal(a, b):
            return False
    return True


__all__ = [
    "PICKLE_PROTOCOL",
    "Artifact",
    "ArtifactIntegrityError",
    "deserialize",
    "load_artifact",
    "round_trip_is_exact",
    "serialize",
    "write_artifact",
]
