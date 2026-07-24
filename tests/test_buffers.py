from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT
from acp_single_pc_deploy.robot.buffers import BufferNotReady, TimedRingBuffer, build_observation


def test_buffer_samples_exact_horizon_and_stride() -> None:
    buffer = TimedRingBuffer(32)
    for index in range(12):
        buffer.append(index * 0.01, np.array([index], dtype=np.float64))
    timestamps, values = buffer.sample_latest(horizon=3, stride=5, now_s=0.12, max_age_s=0.02)
    np.testing.assert_array_equal(timestamps, [0.01, 0.06, 0.11])
    np.testing.assert_array_equal(values[:, 0], [1, 6, 11])


def test_buffer_rejects_non_monotonic_and_stale_samples() -> None:
    buffer = TimedRingBuffer(4)
    buffer.append(1.0, np.zeros(1))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(1.0, np.ones(1))
    with pytest.raises(BufferNotReady, match="stale"):
        buffer.sample_latest(1, 1, now_s=2.0, max_age_s=0.1)


def test_build_observation_keeps_raw_wrench() -> None:
    rgb_buffer = TimedRingBuffer(32)
    pose_buffer = TimedRingBuffer(32)
    wrench_buffer = TimedRingBuffer(64)
    for index in range(32):
        timestamp = 1.0 + index * 0.005
        wrench_buffer.append(timestamp, np.full(6, index, dtype=np.float64))
    for index in range(11):
        timestamp = 1.0 + index * 0.01
        pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        pose_buffer.append(timestamp, pose)
        rgb_buffer.append(timestamp, np.zeros((224, 224, 3), dtype=np.uint8))
    packet = build_observation(
        request_id=9,
        contract=EXPECTED_CONTRACT,
        now_s=1.16,
        rgb_buffer=rgb_buffer,
        pose_buffer=pose_buffer,
        wrench_buffer=wrench_buffer,
        max_age_s={"rgb": 0.1, "pose": 0.1, "wrench": 0.02},
    )
    np.testing.assert_array_equal(packet.wrench[:, 0], np.arange(32))
