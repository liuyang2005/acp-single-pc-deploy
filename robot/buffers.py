from __future__ import annotations

from collections import deque
from threading import Lock

import numpy as np

from acp_single_pc_deploy.common.schemas import ModelContract, ObservationPacket


class BufferNotReady(RuntimeError):
    pass


class TimedRingBuffer:
    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._items: deque[tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self._lock = Lock()

    def append(self, timestamp_s: float, value: np.ndarray) -> None:
        timestamp = float(timestamp_s)
        array = np.asarray(value).copy()
        if not np.isfinite(timestamp) or not np.all(np.isfinite(array)):
            raise ValueError("buffer samples must be finite")
        with self._lock:
            if self._items and timestamp <= self._items[-1][0]:
                raise ValueError("timestamps must be strictly increasing")
            self._items.append((timestamp, array))

    def sample_at_times(
        self,
        target_times_s: np.ndarray,
        now_s: float,
        max_age_s: float,
        max_time_error_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(target_times_s, dtype=np.float64)
        if (
            targets.ndim != 1
            or targets.size == 0
            or not np.all(np.isfinite(targets))
            or np.any(np.diff(targets) <= 0.0)
        ):
            raise ValueError("target times must be a finite, strictly increasing vector")
        if max_time_error_s <= 0.0:
            raise ValueError("max_time_error_s must be positive")
        with self._lock:
            items = list(self._items)
        if not items:
            raise BufferNotReady("buffer is empty")
        available_times = np.asarray([item[0] for item in items], dtype=np.float64)
        selected_indices: list[int] = []
        for target in targets:
            insertion = int(np.searchsorted(available_times, target))
            candidates = [
                index
                for index in (insertion - 1, insertion)
                if 0 <= index < available_times.size
            ]
            selected = min(candidates, key=lambda index: abs(available_times[index] - target))
            error = abs(float(available_times[selected] - target))
            if error > max_time_error_s:
                raise BufferNotReady(
                    f"no sample near target {target:.6f}s: error={error:.6f}s "
                    f"limit={max_time_error_s:.6f}s"
                )
            selected_indices.append(selected)
        if len(set(selected_indices)) != len(selected_indices):
            raise BufferNotReady("target times resolved to duplicate samples")
        timestamps = available_times[selected_indices]
        values = np.stack([items[index][1] for index in selected_indices])
        age = float(now_s) - float(timestamps[-1])
        if age < 0.0:
            raise BufferNotReady(f"latest sample is in the future by {-age:.6f}s")
        if age > max_age_s:
            raise BufferNotReady(f"latest sample is stale: age={age:.6f}s limit={max_age_s:.6f}s")
        return timestamps, values

    def latest(self) -> tuple[float, np.ndarray]:
        with self._lock:
            if not self._items:
                raise BufferNotReady("buffer is empty")
            timestamp, value = self._items[-1]
        return timestamp, value.copy()


def _target_times(
    anchor_time_s: float,
    horizon: int,
    stride: int,
    source_period_s: float,
) -> np.ndarray:
    if horizon <= 0 or stride <= 0 or source_period_s <= 0.0:
        raise ValueError("horizon, stride, and source period must be positive")
    offsets = np.arange(horizon - 1, -1, -1, dtype=np.float64)
    return float(anchor_time_s) - offsets * float(stride) * float(source_period_s)


def build_observation(
    request_id: int,
    contract: ModelContract,
    now_s: float,
    rgb_buffer: TimedRingBuffer,
    pose_buffer: TimedRingBuffer,
    wrench_buffer: TimedRingBuffer,
    max_age_s: dict[str, float],
    source_period_s: dict[str, float],
    max_time_error_s: dict[str, float],
) -> ObservationPacket:
    anchor_time_s, _ = rgb_buffer.latest()
    rgb_ts, rgb = rgb_buffer.sample_at_times(
        _target_times(
            anchor_time_s,
            contract.rgb_horizon,
            contract.rgb_stride,
            source_period_s["rgb"],
        ),
        now_s,
        max_age_s["rgb"],
        max_time_error_s["rgb"],
    )
    pose_ts, pose = pose_buffer.sample_at_times(
        _target_times(
            anchor_time_s,
            contract.pose_horizon,
            contract.pose_stride,
            source_period_s["pose"],
        ),
        now_s,
        max_age_s["pose"],
        max_time_error_s["pose"],
    )
    wrench_ts, wrench = wrench_buffer.sample_at_times(
        _target_times(
            anchor_time_s,
            contract.wrench_horizon,
            contract.wrench_stride,
            source_period_s["wrench"],
        ),
        now_s,
        max_age_s["wrench"],
        max_time_error_s["wrench"],
    )
    packet = ObservationPacket(
        request_id=request_id,
        rgb=rgb,
        pose7=pose,
        wrench=wrench,
        timestamps={"rgb": rgb_ts, "pose": pose_ts, "wrench": wrench_ts},
        latest_age_s={
            "rgb": float(now_s - rgb_ts[-1]),
            "pose": float(now_s - pose_ts[-1]),
            "wrench": float(now_s - wrench_ts[-1]),
        },
    )
    packet.validate(contract)
    return packet
