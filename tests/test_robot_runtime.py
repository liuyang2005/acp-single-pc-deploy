from __future__ import annotations

import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from acp_single_pc_deploy.robot.runner import (
    RobotObservationRuntime,
    should_save_request_frame,
    timing_header,
)
from acp_single_pc_deploy.robot.sensors import resize_rgb_for_policy, write_rgb_png


def _runtime(camera_factory, **overrides) -> RobotObservationRuntime:
    settings = {
        "hardware": None,
        "camera_factory": camera_factory,
        "buffer_capacity": 64,
        "robot_state_hz": 1000.0,
        "warmup_timeout_s": 1.0,
        "max_age_s": {"rgb": 0.2, "pose": 0.05, "wrench": 0.05},
    }
    settings.update(overrides)
    return RobotObservationRuntime(**settings)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("dry-run", True),
        ("execute", False),
        ("continuous-dry-run", True),
        ("continuous", True),
    ],
)
def test_mode_saves_request_frames(mode, expected) -> None:
    assert should_save_request_frame(mode) is expected


def test_continuous_timing_header_is_stable() -> None:
    assert timing_header() == [
        "request_id",
        "chunk_index",
        "inference_latency_s",
        "action_period_s",
        "selected_point_count",
        "command_count",
        "cumulative_runtime_s",
    ]


def test_camera_start_waits_for_first_buffered_frame() -> None:
    read_started = threading.Event()
    release_frame = threading.Event()

    class Source:
        def read(self):
            read_started.set()
            release_frame.wait(timeout=1.0)
            return time.monotonic(), np.zeros((480, 640, 3), dtype=np.uint8)

        def close(self):
            release_frame.set()

    runtime = _runtime(lambda: Source(), camera_start_timeout_s=1.0)
    starter = threading.Thread(target=runtime.start)
    starter.start()
    assert read_started.wait(timeout=1.0)
    assert starter.is_alive()

    release_frame.set()
    starter.join(timeout=1.0)
    assert not starter.is_alive()
    runtime.rgb_buffer.latest()
    runtime.close()


def test_camera_start_recreates_source_after_first_frame_failure() -> None:
    attempts = 0

    class Source:
        def __init__(self, should_fail):
            self.should_fail = should_fail

        def read(self):
            if self.should_fail:
                raise RuntimeError("Frame didn't arrive")
            return time.monotonic(), np.zeros((480, 640, 3), dtype=np.uint8)

        def close(self):
            return None

    def factory():
        nonlocal attempts
        attempts += 1
        return Source(should_fail=attempts == 1)

    runtime = _runtime(
        factory,
        camera_start_attempts=2,
        camera_retry_delay_s=0.0,
        camera_start_timeout_s=1.0,
    )
    runtime.start()

    assert attempts == 2
    runtime.rgb_buffer.latest()
    runtime.close()


def test_wrist_rgb_is_center_cropped_to_policy_shape_without_channel_swap() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :80] = [255, 0, 0]
    image[:, 80:560] = [0, 255, 0]
    image[:, 560:] = [0, 0, 255]
    resized = resize_rgb_for_policy(image, width=224, height=224)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8
    assert np.all(resized[112, 112] == [0, 255, 0])
    assert resized[..., 1].mean() > resized[..., 0].mean()
    assert resized[..., 1].mean() > resized[..., 2].mean()


def test_dry_run_png_preserves_rgb_color_order(tmp_path) -> None:
    import cv2

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = [255, 0, 0]
    path = tmp_path / "frame.png"
    write_rgb_png(path, image)
    loaded_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert np.all(loaded_bgr[0, 0] == [0, 0, 255])


def test_observation_runtime_appends_pose_and_wrench_histories() -> None:
    class Hardware:
        def read_state(self):
            return SimpleNamespace(
                timestamp_s=time.monotonic(),
                pose7=np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64),
                raw_wrench_tcp=np.zeros(6, dtype=np.float64),
                fault=False,
                operational=True,
            )

    runtime = RobotObservationRuntime(
        hardware=Hardware(),
        camera_factory=lambda: None,
        buffer_capacity=64,
        robot_state_hz=1000.0,
        warmup_timeout_s=1.0,
        max_age_s={"rgb": 0.2, "pose": 0.05, "wrench": 0.05},
    )
    runtime._thread = object()
    latest = time.monotonic()
    for index in range(11):
        runtime.rgb_buffer.append(
            latest - (10 - index) * 0.01,
            np.zeros((224, 224, 3), dtype=np.uint8),
        )

    packet = runtime.observe(request_id=3)

    assert packet.request_id == 3
    assert packet.pose7.shape == (3, 7)
    assert packet.wrench.shape == (32, 6)
