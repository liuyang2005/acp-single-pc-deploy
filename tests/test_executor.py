from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.common.schemas import ActionChunk
from acp_single_pc_deploy.robot.executor import ActionChunkExecutor, slerp_pose7
from acp_single_pc_deploy.robot.safety import SafetyLimits, SafetySupervisor
from acp_single_pc_deploy.tests.test_common import make_action


def test_slerp_uses_shortest_quaternion_path() -> None:
    start = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    end = np.array([1, 0, 0, -1, 0, 0, 0], dtype=float)
    middle = slerp_pose7(start, end, 0.5)
    np.testing.assert_allclose(middle[:3], [0.5, 0, 0])
    np.testing.assert_allclose(middle[3:], [1, 0, 0, 0])


def test_executor_uses_reference_virtual_direction_and_current_pose() -> None:
    chunk = make_action()
    chunk.reference_pose7[:, :3] = np.array([0.0, 0.0, 0.0])
    chunk.virtual_pose7[:, :3] = np.array([0.01, 0.0, 0.0])
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    current = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    supervisor.latch_start_pose(current)
    executor = ActionChunkExecutor(chunk, start_time_s=10.0, execute_points=12, inner_stiffness=5000.0)
    command = executor.command_at(10.0, current, supervisor)
    assert command.predicted_stiffness == 500.0
    assert command.applied_stiffness == 500.0
    assert command.equivalent_pose7[0] == pytest.approx(0.001)
    assert command.applied_pose7[0] == pytest.approx(0.001)


def test_executor_never_interpolates_beyond_first_twelve_points() -> None:
    chunk = make_action()
    chunk.reference_pose7[:, 0] = np.arange(16) * 0.001
    chunk.virtual_pose7[:, 0] = np.arange(16) * 0.001
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    start = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    supervisor.latch_start_pose(start)
    executor = ActionChunkExecutor(chunk, start_time_s=0.0, execute_points=12, inner_stiffness=5000.0)
    command = executor.command_at(1.799, start, supervisor)
    assert command.virtual_pose7[0] == pytest.approx(0.011)
    assert executor.expired(1.8)
    with pytest.raises(RuntimeError, match="expired"):
        executor.command_at(1.8, start, supervisor)


def test_execute_points_cannot_exceed_action_horizon() -> None:
    with pytest.raises(ValueError, match="execute_points"):
        ActionChunkExecutor(make_action(), 0.0, execute_points=17, inner_stiffness=5000.0)


def test_executor_can_continue_from_a_later_action_point() -> None:
    chunk = make_action()
    chunk.reference_pose7[:, 0] = np.arange(16) * 0.001
    chunk.virtual_pose7[:, 0] = np.arange(16) * 0.001
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    current = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    supervisor.latch_start_pose(current)
    executor = ActionChunkExecutor(
        chunk,
        start_time_s=0.0,
        execute_points=2,
        inner_stiffness=5000.0,
        start_point=12,
    )

    first = executor.command_at(0.0, current, supervisor)
    second = executor.command_at(chunk.action_period_s, current, supervisor)

    assert first.virtual_pose7[0] == pytest.approx(0.012)
    assert second.virtual_pose7[0] == pytest.approx(0.013)


def test_executor_time_scale_slows_interpolation_and_extends_window() -> None:
    chunk = make_action()
    chunk.reference_pose7[:, 0] = np.arange(16) * 0.001
    chunk.virtual_pose7[:, 0] = np.arange(16) * 0.001
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    current = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    supervisor.latch_start_pose(current)
    executor = ActionChunkExecutor(
        chunk,
        start_time_s=0.0,
        execute_points=2,
        inner_stiffness=5000.0,
        time_scale=2.0,
    )

    second = executor.command_at(2.0 * chunk.action_period_s, current, supervisor)

    assert second.virtual_pose7[0] == pytest.approx(0.001)
    assert executor.action_period_s == pytest.approx(2.0 * chunk.action_period_s)
    assert executor.end_time_s == pytest.approx(4.0 * chunk.action_period_s)


@pytest.mark.parametrize("time_scale", (0.5, np.inf, np.nan))
def test_executor_rejects_invalid_time_scale(time_scale) -> None:
    with pytest.raises(ValueError, match="time_scale"):
        ActionChunkExecutor(
            make_action(),
            0.0,
            execute_points=2,
            inner_stiffness=5000.0,
            time_scale=time_scale,
        )


def test_executor_rejects_a_window_past_the_action_horizon() -> None:
    with pytest.raises(ValueError, match="execution window"):
        ActionChunkExecutor(
            make_action(),
            0.0,
            execute_points=2,
            inner_stiffness=5000.0,
            start_point=15,
        )


def test_executor_uses_reference_orientation_for_translation_only_compliance() -> None:
    chunk = make_action()
    chunk.reference_pose7[:, 3:] = np.array([1.0, 0.0, 0.0, 0.0])
    chunk.virtual_pose7[:, 3:] = np.array([2**-0.5, 0.0, 0.0, 2**-0.5])
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    current = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    supervisor.latch_start_pose(current)
    executor = ActionChunkExecutor(
        chunk,
        start_time_s=0.0,
        execute_points=2,
        inner_stiffness=5000.0,
        orientation_source="reference",
    )
    command = executor.command_at(0.0, current, supervisor)
    np.testing.assert_allclose(command.equivalent_pose7[3:], [1, 0, 0, 0])
