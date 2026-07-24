from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SafetyFault(RuntimeError):
    pass


class DeploymentState(str, Enum):
    INIT = "INIT"
    HOMING = "HOMING"
    READY = "READY"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    FAULT = "FAULT"


@dataclass(frozen=True)
class WrenchReading:
    model_wrench: np.ndarray
    delta_wrench: np.ndarray
    baseline_wrench: np.ndarray

    @classmethod
    def from_raw(cls, raw: np.ndarray, baseline: np.ndarray) -> WrenchReading:
        raw_array = np.asarray(raw, dtype=np.float64)
        baseline_array = np.asarray(baseline, dtype=np.float64)
        if raw_array.shape != (6,) or baseline_array.shape != (6,):
            raise ValueError("raw and baseline wrench must have shape (6,)")
        if not np.all(np.isfinite(raw_array)) or not np.all(np.isfinite(baseline_array)):
            raise ValueError("raw and baseline wrench must be finite")
        return cls(raw_array.copy(), raw_array - baseline_array, baseline_array.copy())


@dataclass(frozen=True)
class SafetyLimits:
    max_raw_force_norm_n: float
    max_raw_torque_norm_nm: float
    max_delta_force_norm_n: float
    max_delta_torque_norm_nm: float
    stiffness_min_n_m: float
    stiffness_max_n_m: float
    inner_translation_stiffness_n_m: float
    max_translation_step_m: float
    max_rotation_step_rad: float
    max_workspace_radius_m: float
    max_rgb_age_s: float
    max_pose_age_s: float
    max_wrench_age_s: float

    @classmethod
    def defaults(cls) -> SafetyLimits:
        return cls(
            max_raw_force_norm_n=25.0,
            max_raw_torque_norm_nm=2.0,
            max_delta_force_norm_n=20.0,
            max_delta_torque_norm_nm=1.0,
            stiffness_min_n_m=200.0,
            stiffness_max_n_m=1000.0,
            inner_translation_stiffness_n_m=5000.0,
            max_translation_step_m=0.002,
            max_rotation_step_rad=0.035,
            max_workspace_radius_m=0.08,
            max_rgb_age_s=0.20,
            max_pose_age_s=0.05,
            max_wrench_age_s=0.05,
        )


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion has zero norm")
    return quaternion / norm


def _slerp_quaternion(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    first = _normalize_quaternion(start)
    second = _normalize_quaternion(end)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(first + fraction * (second - first))
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return _normalize_quaternion(
        np.sin((1.0 - fraction) * theta) / sin_theta * first
        + np.sin(fraction * theta) / sin_theta * second
    )


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        self.state = DeploymentState.INIT
        self.reason = "created"
        self._start_pose7: np.ndarray | None = None
        self._last_limit_messages: tuple[str, ...] = ()

    @property
    def last_limit_messages(self) -> tuple[str, ...]:
        return self._last_limit_messages

    def transition(self, target: DeploymentState, reason: str) -> None:
        if self.state in {DeploymentState.HOLD, DeploymentState.FAULT}:
            raise SafetyFault(f"{self.state.value} is latched; restart is required")
        if target is DeploymentState.FAULT:
            self.state = target
            self.reason = reason
            return
        allowed = {
            DeploymentState.INIT: {DeploymentState.HOMING, DeploymentState.HOLD},
            DeploymentState.HOMING: {DeploymentState.READY, DeploymentState.HOLD},
            DeploymentState.READY: {DeploymentState.ARMED, DeploymentState.HOLD},
            DeploymentState.ARMED: {DeploymentState.RUNNING, DeploymentState.HOLD},
            DeploymentState.RUNNING: {DeploymentState.HOLD},
        }
        if target not in allowed[self.state]:
            raise SafetyFault(f"invalid state transition {self.state.value} -> {target.value}")
        self.state = target
        self.reason = reason

    def latch_start_pose(self, pose7: np.ndarray) -> None:
        self._start_pose7 = self._validated_pose(pose7)

    def validate_sensor_ages(self, ages_s: dict[str, float]) -> None:
        limits = {
            "rgb": self.limits.max_rgb_age_s,
            "pose": self.limits.max_pose_age_s,
            "wrench": self.limits.max_wrench_age_s,
        }
        if set(ages_s) != set(limits):
            self.fault("sensor ages must contain rgb, pose, and wrench")
        for name, limit in limits.items():
            age = ages_s[name]
            if not np.isfinite(age) or age < 0.0 or age > limit:
                self.fault(f"{name} sensor stale or invalid: age={age} limit={limit}")

    def validate_wrench(self, reading: WrenchReading) -> None:
        checks = (
            ("raw force", np.linalg.norm(reading.model_wrench[:3]), self.limits.max_raw_force_norm_n),
            ("raw torque", np.linalg.norm(reading.model_wrench[3:]), self.limits.max_raw_torque_norm_nm),
            ("delta force", np.linalg.norm(reading.delta_wrench[:3]), self.limits.max_delta_force_norm_n),
            ("delta torque", np.linalg.norm(reading.delta_wrench[3:]), self.limits.max_delta_torque_norm_nm),
        )
        for name, value, limit in checks:
            if not np.isfinite(value) or value > limit:
                self.fault(f"{name} norm {value:.3f} exceeds {limit:.3f}")

    def validate_stiffness(self, stiffness: float) -> tuple[float, bool]:
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            self.fault("predicted stiffness must be finite and positive")
        applied = float(np.clip(stiffness, self.limits.stiffness_min_n_m, self.limits.stiffness_max_n_m))
        return applied, applied != float(stiffness)

    def limit_pose(self, requested_pose7: np.ndarray, current_pose7: np.ndarray) -> np.ndarray:
        requested = self._validated_pose(requested_pose7)
        current = self._validated_pose(current_pose7)
        if self._start_pose7 is None:
            self.fault("start pose is not latched")
        assert self._start_pose7 is not None
        radius = float(np.linalg.norm(requested[:3] - self._start_pose7[:3]))
        if radius > self.limits.max_workspace_radius_m:
            self.fault(f"requested pose exceeds workspace radius: {radius:.6f}")
        result = requested.copy()
        messages: list[str] = []
        translation = requested[:3] - current[:3]
        distance = float(np.linalg.norm(translation))
        if distance > self.limits.max_translation_step_m:
            result[:3] = current[:3] + translation * (self.limits.max_translation_step_m / distance)
            messages.append("translation_step")
        dot = abs(float(np.dot(current[3:], requested[3:])))
        angle = 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))
        if angle > self.limits.max_rotation_step_rad:
            result[3:] = _slerp_quaternion(
                current[3:], requested[3:], self.limits.max_rotation_step_rad / angle
            )
            messages.append("rotation_step")
        self._last_limit_messages = tuple(messages)
        return result

    def fault(self, reason: str) -> None:
        self.state = DeploymentState.FAULT
        self.reason = str(reason)
        raise SafetyFault(self.reason)

    @staticmethod
    def _validated_pose(pose7: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose7, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise SafetyFault("pose must be finite shape (7,)")
        result = pose.copy()
        result[3:] = _normalize_quaternion(result[3:])
        return result
