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
    execute_points: int = 12
    control_period_s: float = 0.005


class RobotObservationRuntime:
    def __init__(
        self,
        hardware: Any,
        camera_factory: Callable[[], Any],
        buffer_capacity: int,
        robot_state_hz: float,
        warmup_timeout_s: float,
        max_age_s: dict[str, float],
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

    def observe(self, request_id: int) -> ObservationPacket:
        if self._thread is None:
            raise RuntimeError("observation runtime has not been started")
        deadline = time.monotonic() + self.warmup_timeout_s
        while True:
            self._raise_worker_error()
            state = self.hardware.read_state()
            if state.fault or not state.operational:
                raise SafetyFault("robot became faulted or non-operational during observation warmup")
            self.pose_buffer.append(state.timestamp_s, state.pose7)
            self.wrench_buffer.append(state.timestamp_s, state.raw_wrench_tcp)
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
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if mode not in {"dry-run", "execute"}:
            raise ValueError("mode must be dry-run or execute")
        self.mode = mode
        self.hardware = hardware
        self.camera = camera
        self.client = client
        self.observe = observe
        self.confirm = confirm
        self.clock = clock
        self.sleep = sleep
        self.settings = settings
        self.safety = SafetySupervisor(limits or SafetyLimits.defaults())
        self.event_sink = event_sink or (lambda _event: None)
        self.stop_reason = "not_started"
        self.completed_steps = 0
        self.handshake_metadata: dict[str, Any] = {}
        self._baseline = np.zeros(6, dtype=np.float64)
        self._closed = False

    @classmethod
    def for_test(cls, mode: str, components: Any) -> Runner:
        return cls(
            mode=mode,
            hardware=components.hardware,
            camera=components.camera,
            client=components.client,
            observe=components.observe,
            confirm=components.confirm,
            clock=components.clock,
            sleep=components.sleep,
            settings=RunnerSettings(
                baseline_duration_s=0.0,
                baseline_sample_period_s=0.0,
                control_period_s=0.001,
            ),
            event_sink=components.events.append,
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

    def _sample_baseline(self) -> np.ndarray:
        samples: list[np.ndarray] = []
        deadline = self.clock() + self.settings.baseline_duration_s
        while not samples or self.clock() < deadline:
            state = self.hardware.read_state()
            self._validate_robot_state(state)
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
        self._emit(
            "observation",
            request_id=packet.request_id,
            raw_wrench=reading.model_wrench,
            delta_wrench=reading.delta_wrench,
            latest_age_s=packet.latest_age_s,
        )

    def _execute(self, chunk: Any) -> None:
        state = self.hardware.read_state()
        self._validate_robot_state(state)
        self.safety.latch_start_pose(state.pose7)
        executor = ActionChunkExecutor(
            chunk=chunk,
            start_time_s=self.clock(),
            execute_points=self.settings.execute_points,
            inner_stiffness=self.safety.limits.inner_translation_stiffness_n_m,
        )
        self._transition(DeploymentState.RUNNING, "policy motion confirmed")
        while True:
            now = self.clock()
            if executor.expired(now):
                break
            state = self.hardware.read_state()
            self._validate_robot_state(state)
            reading = WrenchReading.from_raw(state.raw_wrench_tcp, self._baseline)
            self.safety.validate_wrench(reading)
            command = executor.command_at(now, state.pose7, self.safety)
            self.hardware.send_pose(command.applied_pose7)
            self.completed_steps += 1
            self._emit(
                "command",
                request_id=chunk.request_id,
                step=self.completed_steps,
                raw_wrench=reading.model_wrench,
                delta_wrench=reading.delta_wrench,
                predicted_stiffness=command.predicted_stiffness,
                applied_stiffness=command.applied_stiffness,
                equivalent_pose7=command.equivalent_pose7,
                applied_pose7=command.applied_pose7,
                safety_messages=command.safety_messages,
            )
            self.sleep(self.settings.control_period_s)

    def _preview(self, chunk: Any) -> None:
        state = self.hardware.read_state()
        self._validate_robot_state(state)
        reading = WrenchReading.from_raw(state.raw_wrench_tcp, self._baseline)
        self.safety.validate_wrench(reading)
        self.safety.latch_start_pose(state.pose7)
        start_time = self.clock()
        executor = ActionChunkExecutor(
            chunk=chunk,
            start_time_s=start_time,
            execute_points=self.settings.execute_points,
            inner_stiffness=self.safety.limits.inner_translation_stiffness_n_m,
        )
        preview_pose = np.asarray(state.pose7, dtype=np.float64).copy()
        translation_limit_count = 0
        rotation_limit_count = 0
        for point in range(self.settings.execute_points):
            preview_time = start_time + point * chunk.action_period_s
            command = executor.command_at(preview_time, preview_pose, self.safety)
            messages = set(command.safety_messages)
            translation_limit_count += int("translation_step" in messages)
            rotation_limit_count += int("rotation_step" in messages)
            self._emit(
                "action_preview_point",
                request_id=chunk.request_id,
                point=point,
                predicted_stiffness=command.predicted_stiffness,
                applied_stiffness=command.applied_stiffness,
                equivalent_pose7=command.equivalent_pose7,
                limited_pose7=command.applied_pose7,
                safety_messages=command.safety_messages,
            )
            if "stiffness_clipped" in messages:
                self.safety.fault(f"dry-run stiffness clipped at preview point {point}")
            preview_pose = command.applied_pose7
        self._emit(
            "action_preview",
            request_id=chunk.request_id,
            point_count=self.settings.execute_points,
            stiffness_clip_count=0,
            translation_limit_count=translation_limit_count,
            rotation_limit_count=rotation_limit_count,
        )

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
            self.handshake_metadata = dict(handshake)
            self._emit("inference_handshake", response=handshake)

            packet = self.observe(0)
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
            self._transition(DeploymentState.ARMED, "first valid action received")

            if self.mode == "dry-run":
                self._preview(chunk)
                self.stop_reason = "dry_run_complete"
                self._transition(DeploymentState.HOLD, self.stop_reason)
                return 0
            if not self.confirm(
                "输入完整机器人序列号并再次确认，允许执行一个 ACP action chunk: "
            ):
                self.stop_reason = "policy_confirmation_rejected"
                self._transition(DeploymentState.HOLD, self.stop_reason)
                return 2

            self._execute(chunk)
            self.stop_reason = "one_chunk_complete"
            self._transition(DeploymentState.HOLD, self.stop_reason)
            result = 0
        except InferenceTimeout as exc:
            self.stop_reason = f"inference_timeout: {exc}"
            if self.safety.state not in {DeploymentState.HOLD, DeploymentState.FAULT}:
                self._transition(DeploymentState.HOLD, self.stop_reason)
            self._emit("exception", error_type=type(exc).__name__, message=str(exc))
        except Exception as exc:
            self.stop_reason = f"{type(exc).__name__}: {exc}"
            if self.safety.state not in {DeploymentState.HOLD, DeploymentState.FAULT}:
                try:
                    self.safety.fault(self.stop_reason)
                except SafetyFault:
                    pass
            self._emit("exception", error_type=type(exc).__name__, message=str(exc))
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
        max_workspace_radius_m=float(safety["max_workspace_radius_m"]),
        max_rgb_age_s=float(safety["max_rgb_age_s"]),
        max_pose_age_s=float(safety["max_pose_age_s"]),
        max_wrench_age_s=float(safety["max_wrench_age_s"]),
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
    parser.add_argument("--mode", required=True, choices=("dry-run", "execute"))
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_yaml_mapping(args.config)
    require_keys(
        config,
        ("network", "robot", "camera", "acquisition", "execution", "safety", "logging"),
        "robot config",
    )
    run_dir = create_run_directory(config["logging"]["root"], args.mode.replace("-", "_"))
    events = JsonlWriter(run_dir / "events.jsonl")
    timing_path = run_dir / "timing.csv"
    with timing_path.open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            ["request_id", "inference_latency_s", "action_period_s", "completed_command_steps"]
        )
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
        )
        frames_dir = run_dir / "frames"

        def observe(request_id: int) -> ObservationPacket:
            packet = runtime.observe(request_id)
            if args.mode == "dry-run":
                write_rgb_png(frames_dir / f"request_{request_id:06d}.png", packet.rgb[-1])
            return packet

        def event_sink(event: dict[str, Any]) -> None:
            events.write(event)
            if event.get("type") == "action_chunk":
                with timing_path.open("a", encoding="utf-8", newline="") as stream:
                    csv.writer(stream).writerow(
                        [
                            event.get("request_id"),
                            event.get("inference_latency_s"),
                            event.get("action_period_s"),
                            "",
                        ]
                    )

        runner = Runner(
            mode=args.mode,
            hardware=hardware,
            camera=runtime,
            client=client,
            observe=observe,
            confirm=_confirmation(str(config["robot"]["serial"])),
            settings=settings,
            limits=limits,
            event_sink=event_sink,
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
