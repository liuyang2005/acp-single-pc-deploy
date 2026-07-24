from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT, ObservationPacket
from acp_single_pc_deploy.robot.hardware import RobotSample
from acp_single_pc_deploy.tests.test_common import make_action, make_observation


class FakeHardware:
    def __init__(self) -> None:
        self.home_calls: list[np.ndarray] = []
        self.policy_pose_commands: list[np.ndarray] = []
        self.stopped = False
        self.pose7 = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        self.raw_wrench = np.zeros(6, dtype=np.float64)

    def home(self, joints_deg, timeout_s, epsilon_deg):
        self.home_calls.append(np.asarray(joints_deg, dtype=np.float64).copy())
        return {"final_error_deg": 0.0}

    def read_state(self) -> RobotSample:
        return RobotSample(
            timestamp_s=0.0,
            pose7=self.pose7.copy(),
            joints=np.zeros(7),
            raw_wrench_tcp=self.raw_wrench.copy(),
            fault=False,
            operational=True,
        )

    def send_pose(self, pose7) -> None:
        command = np.asarray(pose7, dtype=np.float64).copy()
        self.policy_pose_commands.append(command)
        self.pose7 = command

    def stop(self) -> None:
        self.stopped = True


class FakeCamera:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.inferred_packets: list[ObservationPacket] = []

    def handshake(self):
        return {"contract": EXPECTED_CONTRACT.to_dict(), "checkpoint_sha256": "a" * 64}

    def infer(self, packet: ObservationPacket):
        self.inferred_packets.append(packet)
        chunk = make_action(request_id=packet.request_id)
        chunk.action_period_s = 0.001
        return chunk

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeComponents:
    hardware: FakeHardware = field(default_factory=FakeHardware)
    camera: FakeCamera = field(default_factory=FakeCamera)
    client: FakeClient = field(default_factory=FakeClient)
    confirmations: list[bool] = field(default_factory=lambda: [True, True])
    events: list[dict[str, object]] = field(default_factory=list)
    now_s: float = 0.0

    def confirm(self, _prompt: str) -> bool:
        return self.confirmations.pop(0)

    def observe(self, request_id: int) -> ObservationPacket:
        packet = make_observation(request_id=request_id)
        packet.wrench[:] = self.hardware.raw_wrench
        return packet

    def clock(self) -> float:
        self.now_s += 0.001
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.now_s += max(float(seconds), 0.001)


@pytest.fixture
def fake_components() -> FakeComponents:
    return FakeComponents()
