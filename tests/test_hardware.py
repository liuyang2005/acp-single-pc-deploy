from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from acp_single_pc_deploy.robot import sensors
from acp_single_pc_deploy.robot.hardware import FlexivConfig, FlexivHardware, HOME_JOINTS_DEG
from acp_single_pc_deploy.robot.sensors import RealSenseWristSource, bgr_to_rgb


class FakeTool:
    def __init__(self, robot):
        self.robot = robot

    def exist(self, name):
        return name == "hapticexoteleop"

    def Switch(self, name):
        self.robot.calls.append(("Tool.Switch", name))


class FakeRobot:
    def __init__(self, serial):
        self.serial = serial
        self.calls = []
        self.q = np.zeros(7)
        self.stopped = False

    def fault(self): return False
    def ClearFault(self): self.calls.append(("ClearFault",)); return True
    def Enable(self): self.calls.append(("Enable",))
    def operational(self): return True
    def SwitchMode(self, mode): self.calls.append(("SwitchMode", mode))
    def info(self):
        return SimpleNamespace(q_min=np.full(7, -3.0), q_max=np.full(7, 3.0), K_x_nom=np.array([6000, 6000, 6000, 200, 200, 200]))
    def states(self):
        return SimpleNamespace(
            q=self.q.copy(),
            tcp_pose=np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
            ext_wrench_in_tcp=np.array([1, 2, 3, 0.1, 0.2, 0.3]),
        )
    def SendJointPosition(self, target, velocity, max_velocity, max_acceleration):
        self.calls.append(("SendJointPosition", target, velocity, max_velocity, max_acceleration))
        self.q = np.asarray(target)
    def SetForceControlAxis(self, axes): self.calls.append(("SetForceControlAxis", axes))
    def SetCartesianImpedance(self, stiffness): self.calls.append(("SetCartesianImpedance", stiffness))
    def SetMaxContactWrench(self, wrench): self.calls.append(("SetMaxContactWrench", wrench))
    def SendCartesianMotionForce(self, pose, wrench, **kwargs): self.calls.append(("SendCartesianMotionForce", pose, wrench, kwargs))
    def Stop(self): self.calls.append(("Stop",)); self.stopped = True


class FakeRdk:
    __version__ = "1.9.0"
    Mode = SimpleNamespace(IDLE="IDLE", NRT_JOINT_IMPEDANCE="JOINT", NRT_CARTESIAN_MOTION_FORCE="CART")
    Tool = FakeTool

    def __init__(self):
        self.robot = None

    def Robot(self, serial):
        self.robot = FakeRobot(serial)
        return self.robot


def test_homing_uses_fixed_joint_target_and_expected_mode_order() -> None:
    rdk = FakeRdk()
    hardware = FlexivHardware.connect(FlexivConfig.defaults(), rdk_module=rdk)
    info = hardware.home(HOME_JOINTS_DEG, timeout_s=1.0, epsilon_deg=0.1)
    np.testing.assert_allclose(rdk.robot.q, np.deg2rad(HOME_JOINTS_DEG))
    modes = [call[1] for call in rdk.robot.calls if call[0] == "SwitchMode"]
    assert modes[-2:] == ["JOINT", "IDLE"]
    assert info["final_error_deg"] <= 0.1
    assert not any(call[0] == "ExecutePrimitive" for call in rdk.robot.calls)


def test_cartesian_send_uses_zero_target_wrench() -> None:
    rdk = FakeRdk()
    hardware = FlexivHardware.connect(FlexivConfig.defaults(), rdk_module=rdk)
    hardware.send_pose(np.array([0.4, 0, 0.3, 1, 0, 0, 0], dtype=float))
    sent = [call for call in rdk.robot.calls if call[0] == "SendCartesianMotionForce"][-1]
    assert sent[2] == [0.0] * 6


def test_rdk_version_must_be_1_9() -> None:
    rdk = FakeRdk()
    rdk.__version__ = "1.8.0"
    with pytest.raises(RuntimeError, match="1.9"):
        FlexivHardware.connect(FlexivConfig.defaults(), rdk_module=rdk)


def test_camera_requires_explicit_serial() -> None:
    with pytest.raises(ValueError, match="serial"):
        RealSenseWristSource(serial="", width=640, height=480, fps=30)


def test_bgr_is_converted_to_rgb() -> None:
    bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)
    np.testing.assert_array_equal(bgr_to_rgb(bgr), [[[3, 2, 1]]])


def test_camera_passes_configured_timeout_to_frame_wait(monkeypatch) -> None:
    class Pipeline:
        def __init__(self):
            self.timeout_ms = None

        def start(self, config):
            return None

        def wait_for_frames(self, timeout_ms):
            self.timeout_ms = timeout_ms
            frame = SimpleNamespace(get_data=lambda: np.zeros((480, 640, 3), dtype=np.uint8))
            return SimpleNamespace(get_color_frame=lambda: frame)

        def stop(self):
            return None

    pipeline = Pipeline()
    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: SimpleNamespace(
            enable_device=lambda serial: None,
            enable_stream=lambda *args: None,
        ),
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
    )
    monkeypatch.setattr(sensors.importlib, "import_module", lambda name: fake_rs)

    source = RealSenseWristSource(
        serial="260322274925",
        width=640,
        height=480,
        fps=30,
        frame_timeout_ms=15000,
    )
    source.read()

    assert pipeline.timeout_ms == 15000
