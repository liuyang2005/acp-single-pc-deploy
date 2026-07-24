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

    def sample_latest(
        self,
        horizon: int,
        stride: int,
        now_s: float,
        max_age_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if horizon <= 0 or stride <= 0:
            raise ValueError("horizon and stride must be positive")
        required = (horizon - 1) * stride + 1
        with self._lock:
            items = list(self._items)
        if len(items) < required:
            raise BufferNotReady(f"need {required} samples, have {len(items)}")
        selected = items[-required::stride]
        timestamps = np.asarray([item[0] for item in selected], dtype=np.float64)
        values = np.stack([item[1] for item in selected])
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


def build_observation(
    request_id: int,
    contract: ModelContract,
    now_s: float,
    rgb_buffer: TimedRingBuffer,
    pose_buffer: TimedRingBuffer,
    wrench_buffer: TimedRingBuffer,
    max_age_s: dict[str, float],
) -> ObservationPacket:
    rgb_ts, rgb = rgb_buffer.sample_latest(
        contract.rgb_horizon, contract.rgb_stride, now_s, max_age_s["rgb"]
    )
    pose_ts, pose = pose_buffer.sample_latest(
        contract.pose_horizon, contract.pose_stride, now_s, max_age_s["pose"]
    )
    wrench_ts, wrench = wrench_buffer.sample_latest(
        contract.wrench_horizon, contract.wrench_stride, now_s, max_age_s["wrench"]
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
