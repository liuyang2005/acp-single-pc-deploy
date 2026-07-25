from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acp_single_pc_deploy.common.schemas import ActionChunk, EXPECTED_CONTRACT
from acp_single_pc_deploy.robot.safety import SafetySupervisor, _slerp_quaternion


def slerp_pose7(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    if first.shape != (7,) or second.shape != (7,):
        raise ValueError("poses must have shape (7,)")
    position = first[:3] + fraction * (second[:3] - first[:3])
    quaternion = _slerp_quaternion(first[3:], second[3:], fraction)
    return np.concatenate((position, quaternion))


def compliant_stiffness_matrix(
    reference_position: np.ndarray,
    virtual_position: np.ndarray,
    low_stiffness: float,
    high_stiffness: float,
) -> np.ndarray:
    reference = np.asarray(reference_position, dtype=np.float64)
    virtual = np.asarray(virtual_position, dtype=np.float64)
    direction = virtual - reference
    norm = float(np.linalg.norm(direction))
    axis = np.array([1.0, 0.0, 0.0]) if norm < 1e-3 else direction / norm
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])
    second = np.cross(helper, axis)
    second /= np.linalg.norm(second)
    third = np.cross(axis, second)
    basis = np.column_stack((axis, second, third))
    matrix = basis @ np.diag([low_stiffness, high_stiffness, high_stiffness]) @ basis.T
    return 0.5 * (matrix + matrix.T)


@dataclass(frozen=True)
class ExecutedCommand:
    applied_pose7: np.ndarray
    equivalent_pose7: np.ndarray
    reference_pose7: np.ndarray
    virtual_pose7: np.ndarray
    predicted_stiffness: float
    applied_stiffness: float
    stiffness_matrix: np.ndarray
    safety_messages: tuple[str, ...]


class ActionChunkExecutor:
    def __init__(
        self,
        chunk: ActionChunk,
        start_time_s: float,
        execute_points: int,
        inner_stiffness: float,
        orientation_source: str = "reference",
    ) -> None:
        chunk.validate(EXPECTED_CONTRACT)
        if not 1 <= execute_points <= EXPECTED_CONTRACT.action_horizon:
            raise ValueError("execute_points must be within the action horizon")
        if not np.isfinite(start_time_s):
            raise ValueError("start_time_s must be finite")
        if not np.isfinite(inner_stiffness) or inner_stiffness <= 0.0:
            raise ValueError("inner_stiffness must be finite and positive")
        if orientation_source not in {"reference", "virtual", "current"}:
            raise ValueError("orientation_source must be reference, virtual, or current")
        self.chunk = chunk
        self.start_time_s = float(start_time_s)
        self.execute_points = execute_points
        self.inner_stiffness = float(inner_stiffness)
        self.orientation_source = orientation_source
        self.end_time_s = self.start_time_s + execute_points * chunk.action_period_s

    @property
    def request_id(self) -> int:
        return self.chunk.request_id

    def expired(self, now_s: float) -> bool:
        return float(now_s) >= self.end_time_s

    def command_at(
        self,
        now_s: float,
        current_pose7: np.ndarray,
        safety: SafetySupervisor,
    ) -> ExecutedCommand:
        now = float(now_s)
        if now < self.start_time_s:
            raise RuntimeError("action execution has not started")
        if self.expired(now):
            raise RuntimeError("action execution window expired")
        offset = (now - self.start_time_s) / self.chunk.action_period_s
        left = min(int(np.floor(offset)), self.execute_points - 1)
        right = min(left + 1, self.execute_points - 1)
        fraction = float(np.clip(offset - left, 0.0, 1.0))
        reference = slerp_pose7(self.chunk.reference_pose7[left], self.chunk.reference_pose7[right], fraction)
        virtual = slerp_pose7(self.chunk.virtual_pose7[left], self.chunk.virtual_pose7[right], fraction)
        predicted = float(
            self.chunk.stiffness[left]
            + fraction * (self.chunk.stiffness[right] - self.chunk.stiffness[left])
        )
        applied_stiffness, clipped = safety.validate_stiffness(predicted)
        matrix = compliant_stiffness_matrix(
            reference[:3], virtual[:3], applied_stiffness, self.inner_stiffness
        )
        current = np.asarray(current_pose7, dtype=np.float64)
        equivalent_position = current[:3] + (matrix @ (virtual[:3] - current[:3])) / self.inner_stiffness
        orientation = {
            "reference": reference[3:],
            "virtual": virtual[3:],
            "current": current[3:],
        }[self.orientation_source]
        equivalent = np.concatenate((equivalent_position, orientation))
        applied = safety.limit_pose(equivalent, current)
        messages: list[str] = []
        if clipped:
            messages.append("stiffness_clipped")
        messages.extend(safety.last_limit_messages)
        return ExecutedCommand(
            applied_pose7=applied,
            equivalent_pose7=equivalent,
            reference_pose7=reference,
            virtual_pose7=virtual,
            predicted_stiffness=predicted,
            applied_stiffness=applied_stiffness,
            stiffness_matrix=matrix,
            safety_messages=tuple(messages),
        )
