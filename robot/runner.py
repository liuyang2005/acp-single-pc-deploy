from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from acp_single_pc_deploy.common.config import load_yaml_mapping, require_keys
from acp_single_pc_deploy.common.logging_utils import JsonlWriter, create_run_directory, to_jsonable
from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT, ObservationPacket
from acp_single_pc_deploy.robot.buffers import BufferNotReady, TimedRingBuffer, build_observation
from acp_single_pc_deploy.robot.client import InferenceClient, InferenceTimeout
from acp_single_pc_deploy.robot.executor import ActionChunkExecutor
from acp_single_pc_deploy.robot.hardware import FlexivConfig, FlexivHardware, HOME_JOINTS_DEG
from acp_single_pc_deploy.robot.sensors import (
    RealSenseWristSource,
    resize_rgb_for_policy,
    write_rgb_png,
)
from acp_single_pc_deploy.robot.safety import (
    ContinuousWorkspaceLimits,
    DeploymentState,
    SafetyFault,
    SafetyLimits,
    SafetySupervisor,
    WrenchReading,
)


@dataclass(frozen=True)
class RunnerSettings:
    home_timeout_s: float = 60.0
    home_epsilon_deg: float = 0.5
    baseline_duration_s: float = 2.0
    baseline_sample_period_s: float = 0.005
    execute_points: int = 4
    control_period_s: float = 0.005
    continuous_execute_points: int = 2
    continuous_commitment_points: int = EXPECTED_CONTRACT.action_horizon
    continuous_contact_force_threshold_n: float = 5.0
    continuous_min_upward_exit_m: float = 0.03
    continuous_tracking_speed_utilization: float = 0.8
    continuous_max_time_scale: float = 6.0
    continuous_settle_timeout_s: float = 8.0
    continuous_settle_position_tolerance_m: float = 0.01
    continuous_settle_rotation_tolerance_rad: float = 0.10
    max_linear_velocity_m_s: float = 0.02
    max_linear_acceleration_m_s2: float = 0.05
    max_angular_velocity_rad_s: float = 0.05
    max_angular_acceleration_rad_s2: float = 0.05
    max_continuous_runtime_s: float = 120.0
    orientation_source: str = "reference"
    expected_camera_view: str = "wrist"

    def __post_init__(self) -> None:
        if not 1 <= self.execute_points <= EXPECTED_CONTRACT.action_horizon:
            raise ValueError("execute_points must be within the action horizon")
        if not 1 <= self.continuous_execute_points <= EXPECTED_CONTRACT.action_horizon:
            raise ValueError("continuous_execute_points must be within the action horizon")
        if not (
            self.continuous_execute_points
            <= self.continuous_commitment_points
            <= EXPECTED_CONTRACT.action_horizon
        ):
            raise ValueError(
                "continuous_commitment_points must be between "
                "continuous_execute_points and the action horizon"
            )
        if (
            not np.isfinite(self.continuous_contact_force_threshold_n)
            or self.continuous_contact_force_threshold_n <= 0.0
        ):
            raise ValueError(
                "continuous_contact_force_threshold_n must be finite and positive"
            )
        if (
            not np.isfinite(self.continuous_min_upward_exit_m)
            or self.continuous_min_upward_exit_m <= 0.0
        ):
            raise ValueError(
                "continuous_min_upward_exit_m must be finite and positive"
            )
        if not (
            np.isfinite(self.continuous_tracking_speed_utilization)
            and 0.0 < self.continuous_tracking_speed_utilization <= 1.0
        ):
            raise ValueError(
                "continuous_tracking_speed_utilization must be in (0, 1]"
            )
        if (
            not np.isfinite(self.continuous_max_time_scale)
            or self.continuous_max_time_scale < 1.0
        ):
            raise ValueError("continuous_max_time_scale must be at least 1")
        for name in (
            "continuous_settle_timeout_s",
            "continuous_settle_position_tolerance_m",
            "continuous_settle_rotation_tolerance_rad",
            "max_linear_velocity_m_s",
            "max_linear_acceleration_m_s2",
            "max_angular_velocity_rad_s",
            "max_angular_acceleration_rad_s2",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.orientation_source not in {"reference", "virtual", "current"}:
            raise ValueError("orientation_source must be reference, virtual, or current")
        if self.expected_camera_view not in {"wrist", "main"}:
            raise ValueError("expected_camera_view must be wrist or main")
        if (
            not np.isfinite(self.max_continuous_runtime_s)
            or self.max_continuous_runtime_s <= 0.0
        ):
            raise ValueError("max_continuous_runtime_s must be finite and positive")


MODES = ("dry-run", "execute", "continuous-dry-run", "continuous")


def should_save_request_frame(mode: str) -> bool:
    return mode in {"dry-run", "continuous-dry-run", "continuous"}


def timing_header() -> list[str]:
    return [
        "request_id",
        "chunk_index",
        "inference_latency_s",
        "action_period_s",
        "execution_time_scale",
        "effective_action_period_s",
        "selected_point_count",
        "command_count",
        "cumulative_runtime_s",
    ]


class RobotObservationRuntime:
    def __init__(
        self,
        hardware: Any,
        camera_factory: Callable[[], Any],
        buffer_capacity: int,
        robot_state_hz: float,
        warmup_timeout_s: float,
        max_age_s: dict[str, float],
        source_period_s: dict[str, float],
        max_sample_time_error_factor: float,
        camera_start_attempts: int = 2,
        camera_retry_delay_s: float = 1.0,
        camera_start_timeout_s: float = 35.0,
        policy_width: int = 224,
        policy_height: int = 224,
    ) -> None:
        self.hardware = hardware
        self.camera_factory = camera_factory
        self.robot_period_s = 1.0 / float(robot_state_hz)
        self.warmup_timeout_s = float(warmup_timeout_s)
        self.max_age_s = max_age_s
        if set(source_period_s) != {"rgb", "pose", "wrench"}:
            raise ValueError("source periods must contain rgb, pose, and wrench")
        self.source_period_s = {
            name: float(period) for name, period in source_period_s.items()
        }
        if any(period <= 0.0 for period in self.source_period_s.values()):
            raise ValueError("source periods must be positive")
        if max_sample_time_error_factor <= 0.0:
            raise ValueError("max sample time error factor must be positive")
        self.max_time_error_s = {
            name: float(max_sample_time_error_factor) * period
            for name, period in self.source_period_s.items()
        }
        self.camera_start_attempts = int(camera_start_attempts)
        self.camera_retry_delay_s = float(camera_retry_delay_s)
        self.camera_start_timeout_s = float(camera_start_timeout_s)
        if self.camera_start_attempts <= 0:
            raise ValueError("camera_start_attempts must be positive")
        if self.camera_retry_delay_s < 0.0:
            raise ValueError("camera_retry_delay_s must be nonnegative")
        if self.camera_start_timeout_s <= 0.0:
            raise ValueError("camera_start_timeout_s must be positive")
        self.policy_width = int(policy_width)
        self.policy_height = int(policy_height)
        self.rgb_buffer = TimedRingBuffer(buffer_capacity)
        self.pose_buffer = TimedRingBuffer(buffer_capacity)
        self.wrench_buffer = TimedRingBuffer(buffer_capacity)
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: Any | None = None
        self._worker_error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._camera_loop, name="wrist-rgb", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=self.camera_start_timeout_s):
            raise RuntimeError(
                f"wrist camera did not deliver its first frame within "
                f"{self.camera_start_timeout_s:g} seconds"
            )
        self._raise_worker_error()

    def _camera_loop(self) -> None:
        for attempt in range(1, self.camera_start_attempts + 1):
            source = None
            try:
                source = self.camera_factory()
                self._source = source
                timestamp, frame = self._source.read()
                frame_224 = resize_rgb_for_policy(
                    frame, width=self.policy_width, height=self.policy_height
                )
                self.rgb_buffer.append(timestamp, frame_224)
                self._started.set()
                while not self._stop.is_set():
                    timestamp, frame = self._source.read()
                    frame_224 = resize_rgb_for_policy(
                        frame, width=self.policy_width, height=self.policy_height
                    )
                    self.rgb_buffer.append(timestamp, frame_224)
                return
            except BaseException as exc:
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass
                self._source = None
                if self._stop.is_set():
                    return
                if attempt < self.camera_start_attempts:
                    if self._stop.wait(self.camera_retry_delay_s):
                        return
                    continue
                self._worker_error = exc
                self._started.set()
                return

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(
                f"wrist camera worker failed: {type(self._worker_error).__name__}: {self._worker_error}"
            ) from self._worker_error

    def append_robot_state(self, state: Any) -> None:
        self.pose_buffer.append(state.timestamp_s, state.pose7)
        self.wrench_buffer.append(state.timestamp_s, state.raw_wrench_tcp)

    def observe(self, request_id: int) -> ObservationPacket:
        if self._thread is None:
            raise RuntimeError("observation runtime has not been started")
        deadline = time.monotonic() + self.warmup_timeout_s
        while True:
            self._raise_worker_error()
            state = self.hardware.read_state()
            if state.fault or not state.operational:
                raise SafetyFault("robot became faulted or non-operational during observation warmup")
            self.append_robot_state(state)
            now = time.monotonic()
            try:
                return build_observation(
                    request_id=request_id,
                    contract=EXPECTED_CONTRACT,
                    now_s=now,
                    rgb_buffer=self.rgb_buffer,
                    pose_buffer=self.pose_buffer,
                    wrench_buffer=self.wrench_buffer,
                    max_age_s=self.max_age_s,
                    source_period_s=self.source_period_s,
                    max_time_error_s=self.max_time_error_s,
                )
            except BufferNotReady as exc:
                if now >= deadline:
                    raise RuntimeError(f"observation warmup timed out: {exc}") from exc
                time.sleep(self.robot_period_s)

    def close(self) -> None:
        self._stop.set()
        source = self._source
        if source is not None:
            source.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("wrist camera worker did not stop within 2 seconds")


class Runner:
    def __init__(
        self,
        mode: str,
        hardware: Any,
        camera: Any,
        client: Any,
        observe: Callable[[int], ObservationPacket],
        confirm: Callable[[str], bool],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        settings: RunnerSettings = RunnerSettings(),
        limits: SafetyLimits | None = None,
        continuous_workspace: ContinuousWorkspaceLimits | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        state_sink: Callable[[Any], None] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        is_continuous = mode in {"continuous-dry-run", "continuous"}
        if is_continuous and continuous_workspace is None:
            raise ValueError("continuous modes require continuous workspace limits")
        self.mode = mode
        self.hardware = hardware
        self.camera = camera
        self.client = client
        self.observe = observe
        self.confirm = confirm
        self.clock = clock
        self.sleep = sleep
        self.settings = settings
        self.safety = SafetySupervisor(
            limits or SafetyLimits.defaults(),
            continuous_workspace if is_continuous else None,
        )
        self.event_sink = event_sink or (lambda _event: None)
        self.state_sink = state_sink or (lambda _state: None)
        self.stop_reason = "not_started"
        self.completed_steps = 0
        self.completed_chunks = 0
        self.handshake_metadata: dict[str, Any] = {}
        self._baseline = np.zeros(6, dtype=np.float64)
        self._closed = False
        self._continuous_started_s: float | None = None
        self._continuous_deadline_s: float | None = None
        self._continuous_stop_emitted = False
        self._latest_observation_pose7: np.ndarray | None = None
        self._latest_delta_force_norm_n = 0.0

    @classmethod
    def for_test(
        cls,
        mode: str,
        components: Any,
        settings: RunnerSettings | None = None,
    ) -> Runner:
        test_settings = settings or RunnerSettings(
            baseline_duration_s=0.0,
            baseline_sample_period_s=0.0,
            control_period_s=0.001,
        )
        continuous_workspace = None
        if mode in {"continuous-dry-run", "continuous"}:
            continuous_workspace = ContinuousWorkspaceLimits(
                minimum_xyz_m=np.array([0.55, -0.14, 0.04]),
                maximum_xyz_m=np.array([0.92, 0.13, 0.43]),
                max_equivalent_target_distance_m=0.20,
            )
        return cls(
            mode=mode,
            hardware=components.hardware,
            camera=components.camera,
            client=components.client,
            observe=components.observe,
            confirm=components.confirm,
            clock=components.clock,
            sleep=components.sleep,
            settings=test_settings,
            continuous_workspace=continuous_workspace,
            event_sink=components.events.append,
            state_sink=getattr(components, "record_state", None),
        )

    def _emit(self, event_type: str, **fields: Any) -> None:
        self.event_sink(
            {
                "timestamp_monotonic_s": self.clock(),
                "type": event_type,
                "state": self.safety.state.value,
                **fields,
            }
        )

    def _transition(self, target: DeploymentState, reason: str) -> None:
        source = self.safety.state
        self.safety.transition(target, reason)
        self._emit("state_transition", source=source.value, target=target.value, reason=reason)

    def _record_state(self, state: Any) -> None:
        self.state_sink(state)

    def _sample_baseline(self) -> np.ndarray:
        samples: list[np.ndarray] = []
        deadline = self.clock() + self.settings.baseline_duration_s
        while not samples or self.clock() < deadline:
            state = self.hardware.read_state()
            self._validate_robot_state(state)
            self._record_state(state)
            samples.append(np.asarray(state.raw_wrench_tcp, dtype=np.float64))
            if self.settings.baseline_sample_period_s > 0.0:
                self.sleep(self.settings.baseline_sample_period_s)
        baseline = np.mean(np.stack(samples), axis=0)
        self._emit(
            "wrench_baseline",
            sample_count=len(samples),
            mean=baseline,
            std=np.std(np.stack(samples), axis=0),
        )
        return baseline

    @staticmethod
    def _validate_robot_state(state: Any) -> None:
        if state.fault:
            raise SafetyFault("robot reported fault")
        if not state.operational:
            raise SafetyFault("robot is not operational")

    def _guard_observation(self, packet: ObservationPacket) -> None:
        self.safety.validate_sensor_ages(packet.latest_age_s)
        reading = WrenchReading.from_raw(packet.wrench[-1], self._baseline)
        self.safety.validate_wrench(reading)
        self._latest_observation_pose7 = np.asarray(
            packet.pose7[-1], dtype=np.float64
        ).copy()
        self._latest_delta_force_norm_n = float(
            np.linalg.norm(reading.delta_wrench[:3])
        )
        self._emit(
            "observation",
            request_id=packet.request_id,
            raw_wrench=reading.model_wrench,
            delta_wrench=reading.delta_wrench,
            latest_age_s=packet.latest_age_s,
            sample_timestamps=packet.timestamps,
            history_span_s={
                name: float(times[-1] - times[0])
                for name, times in packet.timestamps.items()
            },
        )

    def _infer_action(self, request_id: int) -> Any:
        packet = self.observe(request_id)
        self._guard_observation(packet)
        chunk = self.client.infer(packet)
        self._emit(
            "action_chunk",
            request_id=chunk.request_id,
            reference_pose7=chunk.reference_pose7,
            virtual_pose7=chunk.virtual_pose7,
            stiffness=chunk.stiffness,
            action_period_s=chunk.action_period_s,
            inference_latency_s=chunk.inference_latency_s,
        )
        return chunk

    def _latch_chunk_pose(self, pose7: np.ndarray) -> None:
        if self.mode in {"continuous-dry-run", "continuous"}:
            self.safety.latch_cycle_pose(pose7)
        else:
            self.safety.latch_start_pose(pose7)

    def _execute(
        self,
        chunk: Any,
        execute_points: int | None = None,
        start_point: int = 0,
        deadline_s: float | None = None,
        time_scale: float = 1.0,
        phase: str = "action",
    ) -> bool:
        state = self.hardware.read_state()
        self._validate_robot_state(state)
        self._record_state(state)
        self._latch_chunk_pose(state.pose7)
        point_count = self.settings.execute_points if execute_points is None else execute_points
        executor = ActionChunkExecutor(
            chunk=chunk,
            start_time_s=self.clock(),
            execute_points=point_count,
            inner_stiffness=self.safety.limits.inner_translation_stiffness_n_m,
            orientation_source=self.settings.orientation_source,
            start_point=start_point,
            time_scale=time_scale,
        )
        while True:
            now = self.clock()
            if deadline_s is not None and now >= deadline_s:
                return False
            if executor.expired(now):
                return True
            state = self.hardware.read_state()
            self._validate_robot_state(state)
            self._record_state(state)
            reading = WrenchReading.from_raw(state.raw_wrench_tcp, self._baseline)
            self.safety.validate_wrench(reading)
            command = executor.command_at(now, state.pose7, self.safety)
            self.hardware.send_pose(command.applied_pose7)
            self.completed_steps += 1
            self._emit(
                "command",
                request_id=chunk.request_id,
                step=self.completed_steps,
                phase=phase,
                plan_start_point=start_point,
                execution_time_scale=time_scale,
                raw_wrench=reading.model_wrench,
                delta_wrench=reading.delta_wrench,
                predicted_stiffness=command.predicted_stiffness,
                applied_stiffness=command.applied_stiffness,
                reference_pose7=command.reference_pose7,
                virtual_pose7=command.virtual_pose7,
                equivalent_pose7=command.equivalent_pose7,
                applied_pose7=command.applied_pose7,
                safety_messages=command.safety_messages,
            )
            self.sleep(self.settings.control_period_s)

    def _preview(
        self,
        chunk: Any,
        execute_points: int | None = None,
        start_point: int = 0,
        deadline_s: float | None = None,
        time_scale: float = 1.0,
    ) -> bool:
        state = self.hardware.read_state()
        self._validate_robot_state(state)
        self._record_state(state)
        reading = WrenchReading.from_raw(state.raw_wrench_tcp, self._baseline)
        self.safety.validate_wrench(reading)
        self._latch_chunk_pose(state.pose7)
        point_count = self.settings.execute_points if execute_points is None else execute_points
        start_time = self.clock()
        executor = ActionChunkExecutor(
            chunk=chunk,
            start_time_s=start_time,
            execute_points=point_count,
            inner_stiffness=self.safety.limits.inner_translation_stiffness_n_m,
            orientation_source=self.settings.orientation_source,
            start_point=start_point,
            time_scale=time_scale,
        )
        preview_pose = np.asarray(state.pose7, dtype=np.float64).copy()
        translation_limit_count = 0
        rotation_limit_count = 0
        completed_points = 0
        for local_point in range(point_count):
            if deadline_s is not None and self.clock() >= deadline_s:
                return False
            point = start_point + local_point
            preview_time = start_time + local_point * executor.action_period_s
            command = executor.command_at(preview_time, preview_pose, self.safety)
            messages = set(command.safety_messages)
            translation_limit_count += int("translation_step" in messages)
            rotation_limit_count += int("rotation_step" in messages)
            self._emit(
                "action_preview_point",
                request_id=chunk.request_id,
                point=point,
                execution_time_scale=time_scale,
                predicted_stiffness=command.predicted_stiffness,
                applied_stiffness=command.applied_stiffness,
                equivalent_pose7=command.equivalent_pose7,
                limited_pose7=command.applied_pose7,
                safety_messages=command.safety_messages,
            )
            if "stiffness_clipped" in messages:
                self.safety.fault(f"dry-run stiffness clipped at preview point {point}")
            preview_pose = command.applied_pose7
            completed_points += 1
        self._emit(
            "action_preview",
            request_id=chunk.request_id,
            point_count=completed_points,
            stiffness_clip_count=0,
            translation_limit_count=translation_limit_count,
            rotation_limit_count=rotation_limit_count,
        )
        return True

    def _begin_continuous(self) -> None:
        self._continuous_started_s = self.clock()
        self._continuous_deadline_s = (
            self._continuous_started_s + self.settings.max_continuous_runtime_s
        )
        self._emit(
            "continuous_start",
            max_runtime_s=self.settings.max_continuous_runtime_s,
            execute_points=self.settings.continuous_execute_points,
            commitment_points=self.settings.continuous_commitment_points,
            contact_force_threshold_n=(
                self.settings.continuous_contact_force_threshold_n
            ),
            min_upward_exit_m=self.settings.continuous_min_upward_exit_m,
            tracking_speed_utilization=(
                self.settings.continuous_tracking_speed_utilization
            ),
            max_time_scale=self.settings.continuous_max_time_scale,
            settle_timeout_s=self.settings.continuous_settle_timeout_s,
            settle_position_tolerance_m=(
                self.settings.continuous_settle_position_tolerance_m
            ),
            settle_rotation_tolerance_rad=(
                self.settings.continuous_settle_rotation_tolerance_rad
            ),
        )

    def _continuous_commitment_metrics(self, chunk: Any) -> tuple[float, float]:
        if self._latest_observation_pose7 is None:
            raise RuntimeError("continuous commitment requires an observation pose")
        final_point = self.settings.continuous_commitment_points - 1
        upward_exit_m = float(
            chunk.virtual_pose7[final_point, 2]
            - self._latest_observation_pose7[2]
        )
        return self._latest_delta_force_norm_n, upward_exit_m

    @staticmethod
    def _quaternion_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
        q1 = np.asarray(first, dtype=np.float64)
        q2 = np.asarray(second, dtype=np.float64)
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = abs(float(np.dot(q1, q2)))
        return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))

    def _continuous_plan_time_scale(
        self, chunk: Any
    ) -> tuple[float, dict[str, float]]:
        point_count = self.settings.continuous_commitment_points
        period_s = float(chunk.action_period_s)
        poses = (
            np.asarray(chunk.reference_pose7[:point_count], dtype=np.float64),
            np.asarray(chunk.virtual_pose7[:point_count], dtype=np.float64),
        )
        linear_velocities: list[np.ndarray] = []
        angular_velocities: list[np.ndarray] = []
        for pose7 in poses:
            linear_velocities.append(np.diff(pose7[:, :3], axis=0) / period_s)
            angular_velocities.append(
                np.asarray(
                    [
                        self._quaternion_angle_rad(left, right) / period_s
                        for left, right in zip(pose7[:-1, 3:], pose7[1:, 3:])
                    ],
                    dtype=np.float64,
                )
            )
        linear_velocity = np.concatenate(linear_velocities, axis=0)
        angular_velocity = np.concatenate(angular_velocities, axis=0)
        max_linear_velocity = float(
            np.max(np.linalg.norm(linear_velocity, axis=1), initial=0.0)
        )
        max_angular_velocity = float(np.max(angular_velocity, initial=0.0))
        linear_acceleration = np.concatenate(
            [np.diff(velocity, axis=0) / period_s for velocity in linear_velocities],
            axis=0,
        )
        angular_acceleration = np.concatenate(
            [np.diff(velocity) / period_s for velocity in angular_velocities],
            axis=0,
        )
        max_linear_acceleration = float(
            np.max(np.linalg.norm(linear_acceleration, axis=1), initial=0.0)
        )
        max_angular_acceleration = float(
            np.max(np.abs(angular_acceleration), initial=0.0)
        )
        utilization = self.settings.continuous_tracking_speed_utilization
        scale_components = {
            "linear_velocity": max_linear_velocity
            / (self.settings.max_linear_velocity_m_s * utilization),
            "linear_acceleration": np.sqrt(
                max_linear_acceleration
                / (self.settings.max_linear_acceleration_m_s2 * utilization)
            ),
            "angular_velocity": max_angular_velocity
            / (self.settings.max_angular_velocity_rad_s * utilization),
            "angular_acceleration": np.sqrt(
                max_angular_acceleration
                / (self.settings.max_angular_acceleration_rad_s2 * utilization)
            ),
        }
        required_scale = float(max(1.0, *scale_components.values()))
        metrics = {
            "max_linear_velocity_m_s": max_linear_velocity,
            "max_linear_acceleration_m_s2": max_linear_acceleration,
            "max_angular_velocity_rad_s": max_angular_velocity,
            "max_angular_acceleration_rad_s2": max_angular_acceleration,
            "required_time_scale": required_scale,
        }
        self._emit(
            "action_plan_time_scaling",
            request_id=chunk.request_id,
            time_scale=required_scale,
            scale_components=scale_components,
            **metrics,
        )
        if required_scale > self.settings.continuous_max_time_scale:
            self.safety.fault(
                "committed action requires time scale "
                f"{required_scale:.3f}, above configured maximum "
                f"{self.settings.continuous_max_time_scale:.3f}"
            )
        return required_scale, metrics

    def _terminal_target_pose(self, chunk: Any, current_pose7: np.ndarray) -> np.ndarray:
        point = self.settings.continuous_commitment_points - 1
        target = np.asarray(chunk.virtual_pose7[point], dtype=np.float64).copy()
        if self.settings.orientation_source == "reference":
            target[3:] = chunk.reference_pose7[point, 3:]
        elif self.settings.orientation_source == "current":
            target[3:] = current_pose7[3:]
        return target

    def _settle_committed_plan(self, chunk: Any, deadline_s: float) -> str:
        final_point = self.settings.continuous_commitment_points - 1
        state = self.hardware.read_state()
        self._validate_robot_state(state)
        self._record_state(state)
        self._latch_chunk_pose(state.pose7)
        target_pose7 = self._terminal_target_pose(chunk, state.pose7)
        started_s = self.clock()
        settle_deadline_s = min(
            deadline_s, started_s + self.settings.continuous_settle_timeout_s
        )
        executor = ActionChunkExecutor(
            chunk=chunk,
            start_time_s=started_s,
            execute_points=1,
            inner_stiffness=self.safety.limits.inner_translation_stiffness_n_m,
            orientation_source=self.settings.orientation_source,
            start_point=final_point,
            time_scale=max(
                1.0,
                (
                    self.settings.continuous_settle_timeout_s
                    + self.settings.control_period_s
                )
                / chunk.action_period_s,
            ),
        )
        self._emit(
            "action_plan_settle_started",
            request_id=chunk.request_id,
            point=final_point,
            target_pose7=target_pose7,
            timeout_s=self.settings.continuous_settle_timeout_s,
        )
        while True:
            now = self.clock()
            state = self.hardware.read_state()
            self._validate_robot_state(state)
            self._record_state(state)
            reading = WrenchReading.from_raw(state.raw_wrench_tcp, self._baseline)
            self.safety.validate_wrench(reading)
            position_error_m = float(
                np.linalg.norm(state.pose7[:3] - target_pose7[:3])
            )
            rotation_error_rad = self._quaternion_angle_rad(
                state.pose7[3:], target_pose7[3:]
            )
            if (
                position_error_m
                <= self.settings.continuous_settle_position_tolerance_m
                and rotation_error_rad
                <= self.settings.continuous_settle_rotation_tolerance_rad
            ):
                self._emit(
                    "action_plan_settle_complete",
                    request_id=chunk.request_id,
                    elapsed_s=now - started_s,
                    position_error_m=position_error_m,
                    rotation_error_rad=rotation_error_rad,
                )
                return "complete"
            if now >= settle_deadline_s:
                reason = (
                    "runtime_limit"
                    if now >= deadline_s
                    else "settle_timeout"
                )
                self._emit(
                    "action_plan_settle_stopped",
                    request_id=chunk.request_id,
                    reason=reason,
                    elapsed_s=now - started_s,
                    position_error_m=position_error_m,
                    rotation_error_rad=rotation_error_rad,
                )
                return reason
            command = executor.command_at(now, state.pose7, self.safety)
            self.hardware.send_pose(command.applied_pose7)
            self.completed_steps += 1
            self._emit(
                "command",
                request_id=chunk.request_id,
                step=self.completed_steps,
                phase="terminal_settle",
                plan_start_point=final_point,
                execution_time_scale=executor.time_scale,
                raw_wrench=reading.model_wrench,
                delta_wrench=reading.delta_wrench,
                predicted_stiffness=command.predicted_stiffness,
                applied_stiffness=command.applied_stiffness,
                reference_pose7=command.reference_pose7,
                virtual_pose7=command.virtual_pose7,
                equivalent_pose7=command.equivalent_pose7,
                applied_pose7=command.applied_pose7,
                position_error_m=position_error_m,
                rotation_error_rad=rotation_error_rad,
                safety_messages=command.safety_messages,
            )
            self.sleep(self.settings.control_period_s)

    def _emit_continuous_stop(self) -> None:
        if self._continuous_started_s is None or self._continuous_stop_emitted:
            return
        self._continuous_stop_emitted = True
        self._emit(
            "continuous_stop",
            stop_reason=self.stop_reason,
            completed_chunks=self.completed_chunks,
            completed_command_steps=self.completed_steps,
            cumulative_runtime_s=self.clock() - self._continuous_started_s,
        )

    def _run_continuous(self, first_chunk: Any) -> int:
        assert self._continuous_started_s is not None
        assert self._continuous_deadline_s is not None
        started_s = self._continuous_started_s
        deadline_s = self._continuous_deadline_s
        next_request_id = first_chunk.request_id + 1
        plan = first_chunk
        plan_start_point = 0
        plan_committed = False
        plan_time_scale = 1.0
        self._transition(DeploymentState.RUNNING, "continuous policy loop started")
        while self.clock() < deadline_s:
            if not plan_committed:
                force_norm_n, upward_exit_m = self._continuous_commitment_metrics(
                    plan
                )
                plan_committed = bool(
                    force_norm_n
                    >= self.settings.continuous_contact_force_threshold_n
                    and upward_exit_m
                    >= self.settings.continuous_min_upward_exit_m
                )
                self._emit(
                    "action_plan_commitment_check",
                    request_id=plan.request_id,
                    contact_force_norm_n=force_norm_n,
                    upward_exit_m=upward_exit_m,
                    committed=plan_committed,
                )
                if plan_committed:
                    plan_time_scale, time_scale_metrics = (
                        self._continuous_plan_time_scale(plan)
                    )
                    self._emit(
                        "action_plan_commitment_started",
                        request_id=plan.request_id,
                        commitment_points=(
                            self.settings.continuous_commitment_points
                        ),
                        contact_force_norm_n=force_norm_n,
                        upward_exit_m=upward_exit_m,
                        execution_time_scale=plan_time_scale,
                        time_scale_metrics=time_scale_metrics,
                    )
            remaining_points = (
                self.settings.continuous_commitment_points - plan_start_point
                if plan_committed
                else self.settings.continuous_execute_points
            )
            point_count = (
                remaining_points
                if plan_committed
                else self.settings.continuous_execute_points
            )
            self._emit(
                "chunk_start",
                request_id=plan.request_id,
                chunk_index=self.completed_chunks,
                plan_start_point=plan_start_point,
                selected_point_count=point_count,
                plan_committed=plan_committed,
                execution_time_scale=plan_time_scale,
                effective_action_period_s=(
                    plan.action_period_s * plan_time_scale
                ),
                cumulative_runtime_s=self.clock() - started_s,
            )
            for point in range(plan_start_point, plan_start_point + point_count):
                self._emit(
                    "action_selected_point",
                    request_id=plan.request_id,
                    point=point,
                    reference_pose7=plan.reference_pose7[point],
                    virtual_pose7=plan.virtual_pose7[point],
                    stiffness=plan.stiffness[point],
                )
            if self.mode == "continuous-dry-run":
                completed = self._preview(
                    plan,
                    execute_points=point_count,
                    start_point=plan_start_point,
                    deadline_s=deadline_s,
                    time_scale=plan_time_scale,
                )
                command_count = 0
            else:
                before = self.completed_steps
                completed = self._execute(
                    plan,
                    execute_points=point_count,
                    start_point=plan_start_point,
                    deadline_s=deadline_s,
                    time_scale=plan_time_scale,
                    phase=(
                        "committed_plan" if plan_committed else "free_space"
                    ),
                )
                command_count = self.completed_steps - before
            if not completed:
                break
            self.completed_chunks += 1
            self._emit(
                "chunk_complete",
                request_id=plan.request_id,
                chunk_index=self.completed_chunks - 1,
                plan_start_point=plan_start_point,
                selected_point_count=point_count,
                command_count=command_count,
                inference_latency_s=plan.inference_latency_s,
                action_period_s=plan.action_period_s,
                execution_time_scale=plan_time_scale,
                effective_action_period_s=(
                    plan.action_period_s * plan_time_scale
                ),
                cumulative_runtime_s=self.clock() - started_s,
            )
            if plan_committed:
                self._emit(
                    "action_plan_commitment_completed",
                    request_id=plan.request_id,
                    completed_commitment_points=(
                        self.settings.continuous_commitment_points
                    ),
                    execution_time_scale=plan_time_scale,
                )
                if self.mode == "continuous-dry-run":
                    self.stop_reason = "dry_run_commitment_complete"
                    self._emit_continuous_stop()
                    self._transition(DeploymentState.HOLD, self.stop_reason)
                    return 0
                settle_result = self._settle_committed_plan(plan, deadline_s)
                if settle_result == "complete":
                    self.stop_reason = "committed_plan_complete"
                    result = 0
                elif settle_result == "runtime_limit":
                    self.stop_reason = "runtime_limit_reached"
                    result = 0
                else:
                    self.stop_reason = "commitment_settle_timeout"
                    result = 1
                self._emit_continuous_stop()
                self._transition(DeploymentState.HOLD, self.stop_reason)
                return result
            if self.clock() >= deadline_s:
                break
            candidate = self._infer_action(next_request_id)
            next_request_id += 1
            self._emit(
                "action_plan_replanned",
                previous_request_id=plan.request_id,
                request_id=candidate.request_id,
            )
            plan = candidate
            plan_start_point = 0
            plan_time_scale = 1.0
        self.stop_reason = "runtime_limit_reached"
        self._emit_continuous_stop()
        self._transition(DeploymentState.HOLD, self.stop_reason)
        return 0

    def run_once(self) -> int:
        result = 1
        try:
            if not self.confirm(
                "确认工作空间已清空并允许机器人自动归位 [y/N]: "
            ):
                self.stop_reason = "homing_confirmation_rejected"
                self._transition(DeploymentState.HOLD, self.stop_reason)
                return 2
            self._transition(DeploymentState.HOMING, "homing confirmed")
            homing = self.hardware.home(
                HOME_JOINTS_DEG,
                timeout_s=self.settings.home_timeout_s,
                epsilon_deg=self.settings.home_epsilon_deg,
            )
            self._transition(DeploymentState.READY, "automatic homing complete")
            self._emit("homing_complete", result=homing)
            start_camera = getattr(self.camera, "start", None)
            if start_camera is not None:
                start_camera()
            self._baseline = self._sample_baseline()
            handshake = self.client.handshake()
            checkpoint_view = str(handshake.get("checkpoint_camera_view", ""))
            if checkpoint_view != self.settings.expected_camera_view:
                raise SafetyFault(
                    f"checkpoint camera view is {checkpoint_view!r}, expected "
                    f"{self.settings.expected_camera_view!r}"
                )
            self.handshake_metadata = dict(handshake)
            self._emit("inference_handshake", response=handshake)

            if self.mode == "continuous-dry-run":
                self._begin_continuous()
                chunk = self._infer_action(0)
                self._transition(DeploymentState.ARMED, "first valid action received")
                return self._run_continuous(chunk)

            chunk = self._infer_action(0)
            self._transition(DeploymentState.ARMED, "first valid action received")

            if self.mode == "dry-run":
                self._preview(chunk)
                self.stop_reason = "dry_run_complete"
                self._transition(DeploymentState.HOLD, self.stop_reason)
                return 0
            execution_name = (
                "连续 ACP 闭环运动" if self.mode == "continuous" else "一个 ACP action chunk"
            )
            if not self.confirm(f"输入完整机器人序列号并再次确认，允许执行{execution_name}: "):
                self.stop_reason = "policy_confirmation_rejected"
                self._transition(DeploymentState.HOLD, self.stop_reason)
                return 2

            if self.mode == "continuous":
                self._begin_continuous()
                return self._run_continuous(chunk)

            self._transition(DeploymentState.RUNNING, "policy motion confirmed")
            self._execute(chunk)
            self.stop_reason = "one_chunk_complete"
            self._transition(DeploymentState.HOLD, self.stop_reason)
            result = 0
        except KeyboardInterrupt:
            self.stop_reason = "operator_interrupt"
            if self.safety.state not in {DeploymentState.HOLD, DeploymentState.FAULT}:
                self._transition(DeploymentState.HOLD, self.stop_reason)
            self._emit_continuous_stop()
            result = 0
        except InferenceTimeout as exc:
            self.stop_reason = f"inference_timeout: {exc}"
            if self.safety.state not in {DeploymentState.HOLD, DeploymentState.FAULT}:
                self._transition(DeploymentState.HOLD, self.stop_reason)
            self._emit("exception", error_type=type(exc).__name__, message=str(exc))
            self._emit_continuous_stop()
        except Exception as exc:
            self.stop_reason = f"{type(exc).__name__}: {exc}"
            if self.safety.state not in {DeploymentState.HOLD, DeploymentState.FAULT}:
                try:
                    self.safety.fault(self.stop_reason)
                except SafetyFault:
                    pass
            self._emit("exception", error_type=type(exc).__name__, message=str(exc))
            self._emit_continuous_stop()
        finally:
            self.close()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for name, resource in (
            ("camera", self.camera),
            ("inference_client", self.client),
            ("hardware", self.hardware),
        ):
            try:
                method = resource.stop if name == "hardware" else resource.close
                method()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        self._emit(
            "shutdown",
            stop_reason=self.stop_reason,
            completed_steps=self.completed_steps,
            completed_chunks=self.completed_chunks,
            cleanup_errors=errors,
        )


def _make_flexiv_config(config: dict[str, Any]) -> FlexivConfig:
    robot = config["robot"]
    execution = config["execution"]
    return FlexivConfig(
        robot_serial=str(robot["serial"]),
        tool_name=str(robot["tool"]),
        enable_timeout_s=float(robot["enable_timeout_s"]),
        clear_fault=bool(robot["clear_fault"]),
        inner_translation_stiffness_n_m=float(execution["inner_translation_stiffness_n_m"]),
        inner_rotation_stiffness_nm_rad=float(execution["inner_rotation_stiffness_nm_rad"]),
        max_contact_wrench=tuple(float(value) for value in execution["max_contact_wrench"]),
        max_linear_velocity_m_s=float(execution["max_linear_velocity_m_s"]),
        max_linear_acceleration_m_s2=float(execution["max_linear_acceleration_m_s2"]),
        max_angular_velocity_rad_s=float(execution["max_angular_velocity_rad_s"]),
        max_angular_acceleration_rad_s2=float(execution["max_angular_acceleration_rad_s2"]),
        home_joint_max_velocity_rad_s=float(robot["home_joint_max_velocity_rad_s"]),
        home_joint_max_acceleration_rad_s2=float(robot["home_joint_max_acceleration_rad_s2"]),
    )


def _make_limits(config: dict[str, Any]) -> SafetyLimits:
    safety = config["safety"]
    execution = config["execution"]
    stiffness_min = float(safety["stiffness_min_n_m"])
    stiffness_max = float(safety["stiffness_max_n_m"])
    inner_stiffness = float(execution["inner_translation_stiffness_n_m"])
    if not 0.0 < stiffness_min <= stiffness_max <= inner_stiffness:
        raise ValueError(
            "policy stiffness range must be positive, ordered, and not exceed "
            "inner translation stiffness"
        )
    return SafetyLimits(
        max_raw_force_norm_n=float(safety["max_raw_force_norm_n"]),
        max_raw_torque_norm_nm=float(safety["max_raw_torque_norm_nm"]),
        max_delta_force_norm_n=float(safety["max_delta_force_norm_n"]),
        max_delta_torque_norm_nm=float(safety["max_delta_torque_norm_nm"]),
        stiffness_min_n_m=stiffness_min,
        stiffness_max_n_m=stiffness_max,
        inner_translation_stiffness_n_m=inner_stiffness,
        max_translation_step_m=float(safety["max_translation_step_m"]),
        max_rotation_step_rad=float(safety["max_rotation_step_rad"]),
        max_equivalent_target_radius_m=float(safety["max_equivalent_target_radius_m"]),
        max_workspace_radius_m=float(safety["max_workspace_radius_m"]),
        max_rgb_age_s=float(safety["max_rgb_age_s"]),
        max_pose_age_s=float(safety["max_pose_age_s"]),
        max_wrench_age_s=float(safety["max_wrench_age_s"]),
    )


def _make_continuous_workspace(config: dict[str, Any]) -> ContinuousWorkspaceLimits:
    continuous = config["continuous"]
    return ContinuousWorkspaceLimits(
        minimum_xyz_m=np.asarray(continuous["workspace_min_xyz_m"], dtype=np.float64),
        maximum_xyz_m=np.asarray(continuous["workspace_max_xyz_m"], dtype=np.float64),
        max_equivalent_target_distance_m=float(
            continuous["max_equivalent_target_distance_m"]
        ),
    )


def _confirmation(serial: str) -> Callable[[str], bool]:
    gate = 0

    def confirm(prompt: str) -> bool:
        nonlocal gate
        gate += 1
        answer = input(prompt).strip()
        if gate == 1:
            return answer.lower() in {"y", "yes"}
        return answer == serial

    return confirm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run guarded single-PC ACP deployment")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_yaml_mapping(args.config)
    require_keys(
        config,
        (
            "network",
            "robot",
            "camera",
            "acquisition",
            "execution",
            "continuous",
            "safety",
            "logging",
        ),
        "robot config",
    )
    run_dir = create_run_directory(config["logging"]["root"], args.mode.replace("-", "_"))
    events = JsonlWriter(run_dir / "events.jsonl")
    timing_path = run_dir / "timing.csv"
    with timing_path.open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(timing_header())
    metadata_path = run_dir / "metadata.json"
    started_metadata = {
        "mode": args.mode,
        "config_path": str(args.config),
        "config": config,
        "status": "starting",
        "started_unix_s": time.time(),
    }
    metadata_path.write_text(
        json.dumps(to_jsonable(started_metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runner: Runner | None = None
    result = 1
    try:
        hardware = FlexivHardware.connect(_make_flexiv_config(config))
        camera = config["camera"]
        acquisition = config["acquisition"]
        limits = _make_limits(config)
        continuous_workspace = (
            _make_continuous_workspace(config)
            if args.mode in {"continuous-dry-run", "continuous"}
            else None
        )
        runtime = RobotObservationRuntime(
            hardware=hardware,
            camera_factory=lambda: RealSenseWristSource(
                serial=str(camera["serial"]),
                width=int(camera["width"]),
                height=int(camera["height"]),
                fps=int(camera["fps"]),
                frame_timeout_ms=int(camera["frame_timeout_ms"]),
            ),
            buffer_capacity=int(acquisition["buffer_capacity"]),
            robot_state_hz=float(acquisition["robot_state_hz"]),
            warmup_timeout_s=float(acquisition["warmup_timeout_s"]),
            camera_start_attempts=int(camera["start_attempts"]),
            camera_retry_delay_s=float(camera["retry_delay_s"]),
            camera_start_timeout_s=float(camera["start_timeout_s"]),
            max_age_s={
                "rgb": limits.max_rgb_age_s,
                "pose": limits.max_pose_age_s,
                "wrench": limits.max_wrench_age_s,
            },
            source_period_s={
                "rgb": 1.0 / float(camera["fps"]),
                "pose": float(acquisition["pose_sample_period_s"]),
                "wrench": float(acquisition["wrench_sample_period_s"]),
            },
            max_sample_time_error_factor=float(
                acquisition["max_sample_time_error_factor"]
            ),
            policy_width=int(camera["policy_width"]),
            policy_height=int(camera["policy_height"]),
        )
        client = InferenceClient(
            str(config["network"]["inference_endpoint"]),
            float(config["network"]["timeout_s"]),
        )
        settings = RunnerSettings(
            home_timeout_s=float(config["robot"]["home_timeout_s"]),
            home_epsilon_deg=float(config["robot"]["home_epsilon_deg"]),
            baseline_duration_s=float(acquisition["baseline_duration_s"]),
            baseline_sample_period_s=float(acquisition["baseline_sample_period_s"]),
            execute_points=int(config["execution"]["execute_points"]),
            control_period_s=float(config["execution"]["control_period_s"]),
            continuous_execute_points=int(config["continuous"]["execute_points"]),
            continuous_commitment_points=int(
                config["continuous"]["commitment_points"]
            ),
            continuous_contact_force_threshold_n=float(
                config["continuous"]["contact_force_threshold_n"]
            ),
            continuous_min_upward_exit_m=float(
                config["continuous"]["min_upward_exit_m"]
            ),
            continuous_tracking_speed_utilization=float(
                config["continuous"]["tracking_speed_utilization"]
            ),
            continuous_max_time_scale=float(
                config["continuous"]["max_time_scale"]
            ),
            continuous_settle_timeout_s=float(
                config["continuous"]["settle_timeout_s"]
            ),
            continuous_settle_position_tolerance_m=float(
                config["continuous"]["settle_position_tolerance_m"]
            ),
            continuous_settle_rotation_tolerance_rad=float(
                config["continuous"]["settle_rotation_tolerance_rad"]
            ),
            max_linear_velocity_m_s=float(
                config["execution"]["max_linear_velocity_m_s"]
            ),
            max_linear_acceleration_m_s2=float(
                config["execution"]["max_linear_acceleration_m_s2"]
            ),
            max_angular_velocity_rad_s=float(
                config["execution"]["max_angular_velocity_rad_s"]
            ),
            max_angular_acceleration_rad_s2=float(
                config["execution"]["max_angular_acceleration_rad_s2"]
            ),
            max_continuous_runtime_s=float(config["continuous"]["max_runtime_s"]),
            orientation_source=str(config["execution"]["orientation_source"]),
            expected_camera_view=str(camera["view"]),
        )
        frames_dir = run_dir / "frames"

        def observe(request_id: int) -> ObservationPacket:
            packet = runtime.observe(request_id)
            if should_save_request_frame(args.mode):
                write_rgb_png(frames_dir / f"request_{request_id:06d}.png", packet.rgb[-1])
            return packet

        def event_sink(event: dict[str, Any]) -> None:
            events.write(event)
            row: list[Any] | None = None
            if event.get("type") == "chunk_complete":
                row = [
                    event.get("request_id"),
                    event.get("chunk_index"),
                    event.get("inference_latency_s"),
                    event.get("action_period_s"),
                    event.get("execution_time_scale"),
                    event.get("effective_action_period_s"),
                    event.get("selected_point_count"),
                    event.get("command_count"),
                    event.get("cumulative_runtime_s"),
                ]
            elif event.get("type") == "action_chunk" and args.mode in {"dry-run", "execute"}:
                row = [
                    event.get("request_id"),
                    "",
                    event.get("inference_latency_s"),
                    event.get("action_period_s"),
                    1.0,
                    event.get("action_period_s"),
                    settings.execute_points,
                    "",
                    "",
                ]
            if row is not None:
                with timing_path.open("a", encoding="utf-8", newline="") as stream:
                    csv.writer(stream).writerow(row)

        runner = Runner(
            mode=args.mode,
            hardware=hardware,
            camera=runtime,
            client=client,
            observe=observe,
            confirm=_confirmation(str(config["robot"]["serial"])),
            settings=settings,
            limits=limits,
            continuous_workspace=continuous_workspace,
            event_sink=event_sink,
            state_sink=runtime.append_robot_state,
        )
        result = runner.run_once()
    except Exception as exc:
        events.write({"type": "startup_exception", "error_type": type(exc).__name__, "message": str(exc)})
    finally:
        final_metadata = {
            **started_metadata,
            "status": "complete" if result == 0 else "stopped",
            "finished_unix_s": time.time(),
            "return_code": result,
            "stop_reason": runner.stop_reason if runner is not None else "startup_failed",
            "completed_command_steps": runner.completed_steps if runner is not None else 0,
            "completed_chunks": runner.completed_chunks if runner is not None else 0,
            "wrench_baseline": runner._baseline if runner is not None else None,
            "inference_handshake": runner.handshake_metadata if runner is not None else None,
        }
        metadata_path.write_text(
            json.dumps(to_jsonable(final_metadata), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        events.close()
    print(f"run directory: {run_dir}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
