from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ModelContract:
    protocol_version: int
    action_dim: int
    rgb_horizon: int
    rgb_stride: int
    pose_horizon: int
    pose_stride: int
    wrench_horizon: int
    wrench_stride: int
    action_horizon: int
    action_stride: int

    def validate_expected(self) -> None:
        if self != EXPECTED_CONTRACT:
            for name, expected in EXPECTED_CONTRACT.to_dict().items():
                actual = getattr(self, name)
                if actual != expected:
                    raise SchemaError(f"{name} must be {expected}, got {actual}")

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelContract:
        try:
            return cls(**{name: int(value[name]) for name in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid model contract: {exc}") from exc


EXPECTED_CONTRACT = ModelContract(1, 19, 2, 10, 3, 5, 32, 1, 16, 50)


def _finite_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise SchemaError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise SchemaError(f"{name} must contain finite numeric values")
    return array


def _validate_pose_history(name: str, value: np.ndarray, horizon: int) -> None:
    pose = _finite_array(name, value, (horizon, 7))
    norms = np.linalg.norm(pose[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        raise SchemaError(f"{name} quaternion must be normalized wxyz")


@dataclass
class ObservationPacket:
    request_id: int
    rgb: np.ndarray
    pose7: np.ndarray
    wrench: np.ndarray
    timestamps: dict[str, np.ndarray]
    latest_age_s: dict[str, float]

    def validate(self, contract: ModelContract) -> None:
        contract.validate_expected()
        if not isinstance(self.request_id, int) or isinstance(self.request_id, bool) or self.request_id < 0:
            raise SchemaError("request_id must be a nonnegative integer")
        rgb = np.asarray(self.rgb)
        expected_rgb = (contract.rgb_horizon, 224, 224, 3)
        if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
            raise SchemaError(f"rgb must be uint8 with shape {expected_rgb}, got {rgb.shape}/{rgb.dtype}")
        _validate_pose_history("pose7", self.pose7, contract.pose_horizon)
        _finite_array("wrench", self.wrench, (contract.wrench_horizon, 6))
        expected_names = {"rgb", "pose", "wrench"}
        if set(self.timestamps) != expected_names or set(self.latest_age_s) != expected_names:
            raise SchemaError("timestamps and latest_age_s must contain rgb, pose, and wrench")
        horizons = {
            "rgb": contract.rgb_horizon,
            "pose": contract.pose_horizon,
            "wrench": contract.wrench_horizon,
        }
        for name, horizon in horizons.items():
            times = _finite_array(f"{name} timestamps", self.timestamps[name], (horizon,))
            if horizon > 1 and np.any(np.diff(times) <= 0.0):
                raise SchemaError(f"{name} timestamps must be strictly increasing")
            age = self.latest_age_s[name]
            if isinstance(age, bool) or not isinstance(age, (int, float, np.floating)) or not np.isfinite(age) or age < 0:
                raise SchemaError(f"{name} latest age must be finite and nonnegative")

    def metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "latest_age_s": {name: float(value) for name, value in self.latest_age_s.items()},
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb,
            "pose7": self.pose7,
            "wrench": self.wrench,
            "rgb_timestamps": self.timestamps["rgb"],
            "pose_timestamps": self.timestamps["pose"],
            "wrench_timestamps": self.timestamps["wrench"],
        }

    @classmethod
    def from_wire(cls, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> ObservationPacket:
        try:
            return cls(
                request_id=int(metadata["request_id"]),
                rgb=arrays["rgb"],
                pose7=arrays["pose7"],
                wrench=arrays["wrench"],
                timestamps={
                    "rgb": arrays["rgb_timestamps"],
                    "pose": arrays["pose_timestamps"],
                    "wrench": arrays["wrench_timestamps"],
                },
                latest_age_s=dict(metadata["latest_age_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid observation wire payload: {exc}") from exc


@dataclass
class ActionChunk:
    request_id: int
    reference_pose7: np.ndarray
    virtual_pose7: np.ndarray
    stiffness: np.ndarray
    action_period_s: float
    inference_latency_s: float

    def validate(self, contract: ModelContract) -> None:
        contract.validate_expected()
        if not isinstance(self.request_id, int) or isinstance(self.request_id, bool) or self.request_id < 0:
            raise SchemaError("request_id must be a nonnegative integer")
        _validate_pose_history("reference_pose7", self.reference_pose7, contract.action_horizon)
        _validate_pose_history("virtual_pose7", self.virtual_pose7, contract.action_horizon)
        stiffness = _finite_array("stiffness", self.stiffness, (contract.action_horizon,))
        if np.any(stiffness <= 0.0):
            raise SchemaError("stiffness must be positive")
        for name, value, allow_zero in (
            ("action_period_s", self.action_period_s, False),
            ("inference_latency_s", self.inference_latency_s, True),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
                raise SchemaError(f"{name} must be finite")
            if value < 0.0 or (not allow_zero and value == 0.0):
                raise SchemaError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")

    def metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action_period_s": float(self.action_period_s),
            "inference_latency_s": float(self.inference_latency_s),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "reference_pose7": self.reference_pose7,
            "virtual_pose7": self.virtual_pose7,
            "stiffness": self.stiffness,
        }

    @classmethod
    def from_wire(cls, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> ActionChunk:
        try:
            return cls(
                request_id=int(metadata["request_id"]),
                reference_pose7=arrays["reference_pose7"],
                virtual_pose7=arrays["virtual_pose7"],
                stiffness=arrays["stiffness"],
                action_period_s=float(metadata["action_period_s"]),
                inference_latency_s=float(metadata["inference_latency_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid action wire payload: {exc}") from exc
