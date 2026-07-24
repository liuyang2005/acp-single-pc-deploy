from __future__ import annotations

import importlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


HOME_JOINTS_DEG = np.array([0.0, -32.0, 0.0, 90.0, 0.0, 28.0, 45.0], dtype=np.float64)


@dataclass(frozen=True)
class FlexivConfig:
    robot_serial: str
    tool_name: str
    enable_timeout_s: float
    clear_fault: bool
    inner_translation_stiffness_n_m: float
    inner_rotation_stiffness_nm_rad: float
    max_contact_wrench: tuple[float, float, float, float, float, float]
    max_linear_velocity_m_s: float
    max_linear_acceleration_m_s2: float
    max_angular_velocity_rad_s: float
    max_angular_acceleration_rad_s2: float
    home_joint_max_velocity_rad_s: float
    home_joint_max_acceleration_rad_s2: float

    @classmethod
    def defaults(cls) -> FlexivConfig:
        return cls(
            robot_serial="Rizon4s-063586",
            tool_name="hapticexoteleop",
            enable_timeout_s=30.0,
            clear_fault=False,
            inner_translation_stiffness_n_m=5000.0,
            inner_rotation_stiffness_nm_rad=100.0,
            max_contact_wrench=(25.0, 25.0, 25.0, 2.0, 2.0, 2.0),
            max_linear_velocity_m_s=0.02,
            max_linear_acceleration_m_s2=0.05,
            max_angular_velocity_rad_s=0.05,
            max_angular_acceleration_rad_s2=0.05,
            home_joint_max_velocity_rad_s=0.25,
            home_joint_max_acceleration_rad_s2=0.60,
        )


@dataclass(frozen=True)
class RobotSample:
    timestamp_s: float
    pose7: np.ndarray
    joints: np.ndarray
    raw_wrench_tcp: np.ndarray
    fault: bool
    operational: bool


class FlexivHardware:
    def __init__(self, robot: Any, rdk: Any, config: FlexivConfig, joint_min: np.ndarray, joint_max: np.ndarray) -> None:
        self._robot = robot
        self._rdk = rdk
        self.config = config
        self._joint_min = joint_min
        self._joint_max = joint_max
        self._owner_thread = threading.get_ident()
        self._cartesian = False
        self._stopped = False

    @classmethod
    def connect(cls, config: FlexivConfig, rdk_module: Any | None = None) -> FlexivHardware:
        rdk = rdk_module or importlib.import_module("flexivrdk")
        version = str(getattr(rdk, "__version__", "unknown"))
        if not (version == "1.9" or version.startswith("1.9.")):
            raise RuntimeError(f"Flexiv RDK 1.9.x is required, found {version}")
        robot = rdk.Robot(config.robot_serial)
        if robot.fault():
            if not config.clear_fault:
                raise RuntimeError("robot is in fault; explicit clear-fault authorization is required")
            if not robot.ClearFault():
                raise RuntimeError("failed to clear robot fault")
        robot.Enable()
        deadline = time.monotonic() + config.enable_timeout_s
        while not robot.operational():
            if time.monotonic() >= deadline:
                raise RuntimeError("robot did not become operational before timeout")
            time.sleep(0.1)
        robot.SwitchMode(rdk.Mode.IDLE)
        tool = rdk.Tool(robot)
        if not tool.exist(config.tool_name):
            raise RuntimeError(f"Flexiv tool {config.tool_name!r} does not exist")
        tool.Switch(config.tool_name)
        info = robot.info()
        joint_min = np.asarray(info.q_min, dtype=np.float64)
        joint_max = np.asarray(info.q_max, dtype=np.float64)
        nominal = np.asarray(info.K_x_nom, dtype=np.float64)
        requested = np.array(
            [config.inner_translation_stiffness_n_m] * 3 + [config.inner_rotation_stiffness_nm_rad] * 3
        )
        if joint_min.shape != (7,) or joint_max.shape != (7,):
            raise RuntimeError("robot joint limits must have shape (7,)")
        if nominal.shape != (6,) or np.any(requested > nominal):
            raise RuntimeError("configured Cartesian stiffness exceeds robot nominal stiffness")
        return cls(robot, rdk, config, joint_min, joint_max)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Flexiv RDK access is restricted to the owner thread")
        if self._stopped:
            raise RuntimeError("Flexiv hardware is stopped")

    def read_state(self) -> RobotSample:
        self._assert_owner()
        state = self._robot.states()
        pose = np.asarray(state.tcp_pose, dtype=np.float64)
        joints = np.asarray(state.q, dtype=np.float64)
        wrench = np.asarray(state.ext_wrench_in_tcp, dtype=np.float64)
        if pose.shape != (7,) or joints.shape != (7,) or wrench.shape != (6,):
            raise RuntimeError("invalid Flexiv state shape")
        if not np.all(np.isfinite(np.concatenate((pose, joints, wrench)))):
            raise RuntimeError("Flexiv state contains non-finite values")
        quaternion_norm = np.linalg.norm(pose[3:])
        if not np.isclose(quaternion_norm, 1.0, atol=1e-3):
            raise RuntimeError("Flexiv pose quaternion is not normalized")
        return RobotSample(
            timestamp_s=time.monotonic(),
            pose7=pose.copy(),
            joints=joints.copy(),
            raw_wrench_tcp=wrench.copy(),
            fault=bool(self._robot.fault()),
            operational=bool(self._robot.operational()),
        )

    def home(self, joints_deg: np.ndarray, timeout_s: float, epsilon_deg: float) -> dict[str, float | list[float]]:
        self._assert_owner()
        target_deg = np.asarray(joints_deg, dtype=np.float64)
        if target_deg.shape != (7,) or not np.all(np.isfinite(target_deg)):
            raise ValueError("home target must be finite shape (7,)")
        target = np.deg2rad(target_deg)
        if np.any(target < self._joint_min) or np.any(target > self._joint_max):
            raise RuntimeError("home target exceeds robot joint limits")
        self._robot.SwitchMode(self._rdk.Mode.NRT_JOINT_IMPEDANCE)
        deadline = time.monotonic() + timeout_s
        epsilon = math.radians(epsilon_deg)
        final_error = float("inf")
        zeros = np.zeros(7)
        max_velocity = np.full(7, self.config.home_joint_max_velocity_rad_s)
        max_acceleration = np.full(7, self.config.home_joint_max_acceleration_rad_s2)
        while True:
            current = np.asarray(self._robot.states().q, dtype=np.float64)
            final_error = float(np.max(np.abs(current - target)))
            if final_error <= epsilon:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"automatic homing timed out; max error={math.degrees(final_error):.3f} deg")
            self._robot.SendJointPosition(
                target.tolist(), zeros.tolist(), max_velocity.tolist(), max_acceleration.tolist()
            )
            time.sleep(0.02)
        self._robot.SendJointPosition(
            target.tolist(), zeros.tolist(), max_velocity.tolist(), max_acceleration.tolist()
        )
        self._robot.SwitchMode(self._rdk.Mode.IDLE)
        self._cartesian = False
        return {"target_joints_deg": target_deg.tolist(), "final_error_deg": math.degrees(final_error)}

    def enter_cartesian_control(self) -> None:
        self._assert_owner()
        if self._cartesian:
            return
        self._robot.SwitchMode(self._rdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        self._robot.SetForceControlAxis([False] * 6)
        self._robot.SetCartesianImpedance(
            [self.config.inner_translation_stiffness_n_m] * 3
            + [self.config.inner_rotation_stiffness_nm_rad] * 3
        )
        self._robot.SetMaxContactWrench(list(self.config.max_contact_wrench))
        self._cartesian = True

    def send_pose(self, pose7_wxyz: np.ndarray) -> None:
        self._assert_owner()
        pose = np.asarray(pose7_wxyz, dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("Cartesian pose must be finite shape (7,)")
        norm = float(np.linalg.norm(pose[3:]))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError("Cartesian quaternion must be normalized wxyz")
        self.enter_cartesian_control()
        self._robot.SendCartesianMotionForce(
            pose.tolist(),
            [0.0] * 6,
            max_linear_vel=self.config.max_linear_velocity_m_s,
            max_linear_acc=self.config.max_linear_acceleration_m_s2,
            max_angular_vel=self.config.max_angular_velocity_rad_s,
            max_angular_acc=self.config.max_angular_acceleration_rad_s2,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Flexiv stop must run on owner thread")
        self._stopped = True
        try:
            self._robot.Stop()
        finally:
            try:
                self._robot.SwitchMode(self._rdk.Mode.IDLE)
            except Exception:
                pass
