"""Content-addressed preregistration for A3 shadow models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@dataclass(frozen=True)
class ShadowModelSpec:
    model_id: str
    model_version: str
    baseline: bool
    config: Mapping[str, Any]
    feature_version: str
    training_rule: str
    data_requirements: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShadowModelSpec:
        return cls(
            model_id=str(payload["model_id"]),
            model_version=str(payload["model_version"]),
            baseline=bool(payload["baseline"]),
            config=MappingProxyType(dict(payload["config"])),
            feature_version=str(payload["feature_version"]),
            training_rule=str(payload["training_rule"]),
            data_requirements=tuple(payload["data_requirements"]),
        )


@dataclass(frozen=True)
class ShadowRegistry:
    registry_version: str
    instrument: str
    horizon_bars: int
    feature_contract_version: str
    target_definition_version: str
    models: tuple[ShadowModelSpec, ...]
    schedule: Mapping[str, Any]
    registry_hash: str

    @classmethod
    def load(cls, path: Path | None = None) -> ShadowRegistry:
        if path is None:
            raw = files("btc_forecaster.shadow").joinpath("registry-v1.json").read_bytes()
        else:
            raw = path.read_bytes()
        payload = json.loads(raw)
        canonical = canonical_json(payload)
        if not payload.get("created_before_forward_outcomes"):
            raise ValueError("registry is not preregistered")
        models = tuple(ShadowModelSpec.from_dict(item) for item in payload["models"])
        if not any(model.baseline for model in models):
            raise ValueError("shadow registry requires a baseline")
        return cls(
            registry_version=str(payload["registry_version"]),
            instrument=str(payload["instrument"]),
            horizon_bars=int(payload["horizon_bars"]),
            feature_contract_version=str(payload["feature_contract_version"]),
            target_definition_version=str(payload["target_definition_version"]),
            models=models,
            schedule=MappingProxyType(dict(payload["schedule"])),
            registry_hash=sha256(canonical).hexdigest(),
        )
