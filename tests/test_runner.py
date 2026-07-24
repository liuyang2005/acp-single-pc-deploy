from __future__ import annotations

import numpy as np

from acp_single_pc_deploy.robot.hardware import HOME_JOINTS_DEG
from acp_single_pc_deploy.robot.runner import Runner
from acp_single_pc_deploy.robot.safety import DeploymentState


def test_dry_run_homes_and_infers_without_policy_pose(fake_components) -> None:
    runner = Runner.for_test("dry-run", fake_components)
    assert runner.run_once() == 0
    np.testing.assert_array_equal(fake_components.hardware.home_calls[0], HOME_JOINTS_DEG)
    assert len(fake_components.client.inferred_packets) == 1
    assert fake_components.hardware.policy_pose_commands == []
    assert runner.safety.state is DeploymentState.HOLD


def test_dry_run_previews_every_execution_point(fake_components) -> None:
    runner = Runner.for_test("dry-run", fake_components)

    assert runner.run_once() == 0

    points = [
        event
        for event in fake_components.events
        if event["type"] == "action_preview_point"
    ]
    assert len(points) == runner.settings.execute_points == 12
    assert [event["point"] for event in points] == list(range(12))
    assert fake_components.hardware.policy_pose_commands == []


def test_dry_run_fails_when_any_stiffness_is_clipped(fake_components) -> None:
    original_infer = fake_components.client.infer

    def clipped_infer(packet):
        chunk = original_infer(packet)
        chunk.stiffness[5] = 5001.0
        return chunk

    fake_components.client.infer = clipped_infer
    runner = Runner.for_test("dry-run", fake_components)

    assert runner.run_once() == 1
    assert runner.safety.state is DeploymentState.FAULT
    assert fake_components.hardware.policy_pose_commands == []


def test_dry_run_safety_previews_model_action_without_sending(fake_components) -> None:
    original_infer = fake_components.client.infer

    def unsafe_infer(packet):
        chunk = original_infer(packet)
        chunk.virtual_pose7[:, 0] = 1.0
        return chunk

    fake_components.client.infer = unsafe_infer
    runner = Runner.for_test("dry-run", fake_components)
    assert runner.run_once() == 1
    assert fake_components.hardware.policy_pose_commands == []
    assert runner.safety.state is DeploymentState.FAULT


def test_execute_requires_both_confirmation_gates(fake_components) -> None:
    fake_components.confirmations = [True, False]
    runner = Runner.for_test("execute", fake_components)
    assert runner.run_once() == 2
    assert fake_components.hardware.policy_pose_commands == []
    assert runner.safety.state is DeploymentState.HOLD


def test_execute_preserves_raw_wrench_and_stops_after_one_chunk(fake_components) -> None:
    fake_components.hardware.raw_wrench = np.array([1, 2, 3, 0.1, 0.2, 0.3], dtype=float)
    runner = Runner.for_test("execute", fake_components)
    assert runner.run_once() == 0
    np.testing.assert_array_equal(
        fake_components.client.inferred_packets[0].wrench[-1],
        fake_components.hardware.raw_wrench,
    )
    assert 1 <= len(fake_components.hardware.policy_pose_commands) <= 12
    assert runner.safety.state is DeploymentState.HOLD


def test_shutdown_stops_every_owned_resource(fake_components) -> None:
    runner = Runner.for_test("dry-run", fake_components)
    runner.run_once()
    assert fake_components.camera.closed
    assert fake_components.client.closed
    assert fake_components.hardware.stopped
