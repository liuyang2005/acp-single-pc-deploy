from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT
from acp_single_pc_deploy.robot.buffers import BufferNotReady, TimedRingBuffer, build_observation


def test_buffer_samples_exact_horizon_and_stride() -> None:
    buffer = TimedRingBuffer(32)
    for index in range(12):
        buffer.append(index * 0.01, np.array([index], dtype=np.float64))
    timestamps, values = buffer.sample_at_times(
        np.array([0.01, 0.06, 0.11]),
        now_s=0.12,
        max_age_s=0.02,
        max_time_error_s=0.001,
    )
    np.testing.assert_array_equal(timestamps, [0.01, 0.06, 0.11])
    np.testing.assert_array_equal(values[:, 0], [1, 6, 11])


def test_buffer_rejects_non_monotonic_and_stale_samples() -> None:
    buffer = TimedRingBuffer(4)
    buffer.append(1.0, np.zeros(1))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(1.0, np.ones(1))
    with pytest.raises(BufferNotReady, match="stale"):
        buffer.sample_at_times(
            np.array([1.0]), now_s=2.0, max_age_s=0.1, max_time_error_s=0.01
        )


def test_build_observation_uses_training_time_spans_and_keeps_raw_wrench() -> None:
    rgb_buffer = TimedRingBuffer(8)
    pose_buffer = TimedRingBuffer(8)
    wrench_buffer = TimedRingBuffer(64)
    anchor = 1.2
    rgb_times = np.array([0.9, anchor])
    pose_times = np.array([1.1, 1.15, anchor])
    wrench_times = anchor - np.arange(31, -1, -1) * 0.005
    for timestamp in rgb_times:
        rgb_buffer.append(timestamp, np.zeros((224, 224, 3), dtype=np.uint8))
    pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    for timestamp in pose_times:
        pose_buffer.append(timestamp, pose)
    for index, timestamp in enumerate(wrench_times):
        wrench_buffer.append(timestamp, np.full(6, index, dtype=np.float64))

    packet = build_observation(
        request_id=9,
        contract=EXPECTED_CONTRACT,
        now_s=1.205,
        rgb_buffer=rgb_buffer,
        pose_buffer=pose_buffer,
        wrench_buffer=wrench_buffer,
        max_age_s={"rgb": 0.1, "pose": 0.1, "wrench": 0.02},
        source_period_s={"rgb": 0.03, "pose": 0.01, "wrench": 0.005},
        max_time_error_s={"rgb": 0.001, "pose": 0.001, "wrench": 0.001},
    )

    np.testing.assert_allclose(packet.timestamps["rgb"], rgb_times)
    np.testing.assert_allclose(packet.timestamps["pose"], pose_times)
    np.testing.assert_allclose(packet.timestamps["wrench"], wrench_times)
    np.testing.assert_array_equal(packet.wrench[:, 0], np.arange(32))


def test_buffer_rejects_a_history_gap_even_when_latest_sample_is_fresh() -> None:
    buffer = TimedRingBuffer(4)
    buffer.append(0.0, np.zeros(1))
    buffer.append(1.0, np.ones(1))
    with pytest.raises(BufferNotReady, match="no sample near target"):
        buffer.sample_at_times(
            np.array([0.9, 1.0]),
            now_s=1.0,
            max_age_s=0.1,
            max_time_error_s=0.01,
        )
