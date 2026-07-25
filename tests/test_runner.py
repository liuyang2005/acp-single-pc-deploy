from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.robot.hardware import HOME_JOINTS_DEG
from acp_single_pc_deploy.robot.client import InferenceTimeout
from acp_single_pc_deploy.robot.runner import Runner, RunnerSettings
from acp_single_pc_deploy.robot.safety import DeploymentState


def continuous_settings(**overrides) -> RunnerSettings:
    values = {
        "baseline_duration_s": 0.0,
        "baseline_sample_period_s": 0.0,
        "control_period_s": 0.001,
        "continuous_execute_points": 4,
        "max_continuous_runtime_s": 0.03,
    }
    values.update(overrides)
    return RunnerSettings(**values)


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
    assert len(points) == runner.settings.execute_points == 4
    assert [event["point"] for event in points] == list(
        range(runner.settings.execute_points)
    )
    assert fake_components.hardware.policy_pose_commands == []


def test_dry_run_fails_when_any_stiffness_is_clipped(fake_components) -> None:
    original_infer = fake_components.client.infer

    def clipped_infer(packet):
        chunk = original_infer(packet)
        chunk.stiffness[3] = 5001.0
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
        chunk.virtual_pose7[:, 0] = 3.0
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
    shutdown = [e for e in fake_components.events if e["type"] == "shutdown"]
    assert shutdown[0]["completed_chunks"] == runner.completed_chunks


def test_continuous_executes_four_points_then_reobserves(fake_components) -> None:
    runner = Runner.for_test(
        "continuous",
        fake_components,
        settings=continuous_settings(max_continuous_runtime_s=0.04),
    )

    assert runner.run_once() == 0
    assert fake_components.observed_request_ids == list(
        range(len(fake_components.observed_request_ids))
    )
    assert len(fake_components.observed_request_ids) >= 2
    complete = [e for e in fake_components.events if e["type"] == "chunk_complete"]
    assert all(e["selected_point_count"] == 4 for e in complete)
    starts = [e for e in fake_components.events if e["type"] == "chunk_start"]
    selected = [
        e for e in fake_components.events if e["type"] == "action_selected_point"
    ]
    assert len(selected) == 4 * len(starts)
    assert [e["point"] for e in selected[:8]] == [0, 1, 2, 3] * 2
    assert [e["request_id"] for e in selected[:8]] == [0] * 4 + [1] * 4
    assert runner.completed_chunks == len(complete)


def test_continuous_commits_contact_plan_with_a_clear_upward_exit(
    fake_components,
) -> None:
    original_observe = fake_components.observe
    original_infer = fake_components.client.infer

    def contact_observe(request_id):
        packet = original_observe(request_id)
        packet.wrench[:, 0] = 6.0
        return packet

    def upward_infer(packet):
        chunk = original_infer(packet)
        chunk.virtual_pose7[3, 2] = packet.pose7[-1, 2] + 0.04
        return chunk

    fake_components.observe = contact_observe
    fake_components.client.infer = upward_infer
    runner = Runner.for_test(
        "continuous",
        fake_components,
        settings=continuous_settings(
            continuous_execute_points=2,
            continuous_commitment_points=4,
            max_continuous_runtime_s=0.1,
        ),
    )

    assert runner.run_once() == 0
    selected = [
        (e["request_id"], e["point"])
        for e in fake_components.events
        if e["type"] == "action_selected_point"
    ]
    assert selected[:4] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    committed = [
        e
        for e in fake_components.events
        if e["type"] == "action_plan_commitment_started"
    ]
    assert committed
    assert committed[0]["request_id"] == 0
    assert committed[0]["contact_force_norm_n"] == 6.0
    assert committed[0]["upward_exit_m"] == pytest.approx(0.04)


def test_continuous_contact_without_upward_exit_keeps_replanning(
    fake_components,
) -> None:
    original_observe = fake_components.observe

    def contact_observe(request_id):
        packet = original_observe(request_id)
        packet.wrench[:, 0] = 6.0
        return packet

    fake_components.observe = contact_observe
    runner = Runner.for_test(
        "continuous",
        fake_components,
        settings=continuous_settings(max_continuous_runtime_s=0.05),
    )

    assert runner.run_once() == 0
    assert not any(
        e["type"] == "action_plan_commitment_started"
        for e in fake_components.events
    )
    checks = [
        e
        for e in fake_components.events
        if e["type"] == "action_plan_commitment_check"
    ]
    assert checks
    assert all(e["committed"] is False for e in checks)


def test_continuous_dry_run_repeats_without_sending_pose(fake_components) -> None:
    runner = Runner.for_test(
        "continuous-dry-run", fake_components, settings=continuous_settings()
    )

    assert runner.run_once() == 0
    assert len(fake_components.observed_request_ids) >= 2
    assert fake_components.hardware.policy_pose_commands == []
    assert runner.stop_reason == "runtime_limit_reached"


def test_continuous_interrupt_is_normal_hold_and_cleans_up(fake_components) -> None:
    original_observe = fake_components.observe

    def interrupt_on_second_request(request_id):
        if request_id == 1:
            raise KeyboardInterrupt
        return original_observe(request_id)

    fake_components.observe = interrupt_on_second_request
    runner = Runner.for_test(
        "continuous",
        fake_components,
        settings=continuous_settings(max_continuous_runtime_s=0.1),
    )

    assert runner.run_once() == 0
    assert runner.stop_reason == "operator_interrupt"
    assert runner.safety.state is DeploymentState.HOLD
    assert fake_components.hardware.stopped
    assert any(
        e["type"] == "continuous_stop" and e["stop_reason"] == "operator_interrupt"
        for e in fake_components.events
    )


def test_continuous_inference_timeout_never_reuses_previous_action(fake_components) -> None:
    original_infer = fake_components.client.infer

    def timeout_on_second_request(packet):
        if packet.request_id == 1:
            raise InferenceTimeout("injected")
        return original_infer(packet)

    fake_components.client.infer = timeout_on_second_request
    runner = Runner.for_test(
        "continuous",
        fake_components,
        settings=continuous_settings(max_continuous_runtime_s=0.1),
    )

    assert runner.run_once() == 1
    assert runner.stop_reason.startswith("inference_timeout:")
    assert [p.request_id for p in fake_components.client.inferred_packets] == [0]
    assert runner.completed_chunks == 1


def test_deadline_mid_chunk_does_not_increment_completed_chunks(fake_components) -> None:
    runner = Runner.for_test(
        "continuous",
        fake_components,
        settings=continuous_settings(
            max_continuous_runtime_s=0.03,
            control_period_s=0.02,
        ),
    )

    assert runner.run_once() == 0
    assert runner.completed_chunks == 0
    assert runner.completed_steps > 0


def test_continuous_events_have_one_start_and_stop_summary(fake_components) -> None:
    runner = Runner.for_test(
        "continuous-dry-run", fake_components, settings=continuous_settings()
    )

    assert runner.run_once() == 0
    starts = [e for e in fake_components.events if e["type"] == "continuous_start"]
    stops = [e for e in fake_components.events if e["type"] == "continuous_stop"]
    assert len(starts) == len(stops) == 1
    assert stops[0]["completed_chunks"] == runner.completed_chunks
    assert stops[0]["completed_command_steps"] == runner.completed_steps
    assert "cumulative_runtime_s" in stops[0]


def test_runner_rejects_checkpoint_for_another_camera_view(fake_components) -> None:
    fake_components.client.handshake = lambda: {
        "contract": {},
        "checkpoint_sha256": "a" * 64,
        "checkpoint_camera_view": "main",
    }
    runner = Runner.for_test("dry-run", fake_components)
    assert runner.run_once() == 1
    assert runner.safety.state is DeploymentState.FAULT
    assert "camera view" in runner.stop_reason
