from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from acp_single_pc_deploy.robot.safety import (
    ContinuousWorkspaceLimits,
    DeploymentState,
    SafetyFault,
    SafetyLimits,
    SafetySupervisor,
    WrenchReading,
)


def pose(x=0.0) -> np.ndarray:
    return np.array([x, 0, 0, 1, 0, 0, 0], dtype=np.float64)


def continuous_workspace() -> ContinuousWorkspaceLimits:
    return ContinuousWorkspaceLimits(
        minimum_xyz_m=np.array([0.55, -0.14, 0.04]),
        maximum_xyz_m=np.array([0.92, 0.13, 0.43]),
        max_equivalent_target_distance_m=0.20,
    )


def test_model_wrench_is_raw_while_delta_is_diagnostic() -> None:
    raw = np.array([3, 4, 0, 0.2, 0, 0], dtype=np.float64)
    baseline = np.array([1, 1, 0, 0.1, 0, 0], dtype=np.float64)
    reading = WrenchReading.from_raw(raw, baseline)
    np.testing.assert_array_equal(reading.model_wrench, raw)
    np.testing.assert_allclose(reading.delta_wrench, raw - baseline)


def test_raw_and_delta_wrench_are_both_guarded() -> None:
    limits = SafetyLimits.defaults()
    supervisor = SafetySupervisor(limits)
    with pytest.raises(SafetyFault, match="raw force"):
        supervisor.validate_wrench(WrenchReading.from_raw(np.array([26, 0, 0, 0, 0, 0]), np.zeros(6)))
    supervisor = SafetySupervisor(limits)
    with pytest.raises(SafetyFault, match="delta torque"):
        supervisor.validate_wrench(WrenchReading.from_raw(np.array([0, 0, 0, 0, 1.1, 0]), np.array([0, 0, 0, 0, 0, 0])))


def test_hold_and_fault_are_latched() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    supervisor.transition(DeploymentState.HOMING, "confirmed")
    supervisor.transition(DeploymentState.READY, "homed")
    supervisor.transition(DeploymentState.HOLD, "done")
    with pytest.raises(SafetyFault, match="latched"):
        supervisor.transition(DeploymentState.RUNNING, "resume")


def test_pose_is_step_limited_within_target_and_workspace_guards() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    supervisor.latch_start_pose(pose())
    limited = supervisor.limit_pose(pose(0.09), current_pose7=pose())
    assert limited[0] == pytest.approx(0.002)
    assert "translation_step" in supervisor.last_limit_messages


def test_equivalent_target_radius_faults_before_step_limiting() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    supervisor.latch_start_pose(pose())

    with pytest.raises(SafetyFault, match="equivalent target radius"):
        supervisor.limit_pose(pose(0.201), current_pose7=pose())


def test_step_limited_command_still_obeys_applied_workspace_radius() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    supervisor.latch_start_pose(pose())

    with pytest.raises(SafetyFault, match="applied pose exceeds workspace radius"):
        supervisor.limit_pose(pose(0.15), current_pose7=pose(0.079))


def test_stiffness_is_clipped_and_nonfinite_faults() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    assert supervisor.validate_stiffness(1500.0) == (1000.0, True)
    with pytest.raises(SafetyFault, match="stiffness"):
        supervisor.validate_stiffness(float("nan"))


def test_float32_boundary_noise_is_not_reported_as_stiffness_clipping() -> None:
    limits = replace(SafetyLimits.defaults(), stiffness_max_n_m=5000.0)
    supervisor = SafetySupervisor(limits)

    assert supervisor.validate_stiffness(5000.00048828125) == (5000.0, False)
    assert supervisor.validate_stiffness(5000.01) == (5000.0, True)


def test_continuous_workspace_accepts_pose_inside_absolute_box() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults(), continuous_workspace())
    current = np.array([0.70, 0.00, 0.20, 1, 0, 0, 0], dtype=float)
    supervisor.latch_cycle_pose(current)
    requested = current.copy()
    requested[0] += 0.01

    applied = supervisor.limit_pose(requested, current)

    assert 0.55 <= applied[0] <= 0.92


@pytest.mark.parametrize(
    "axis,value",
    [(0, 0.549), (0, 0.921), (1, -0.141), (1, 0.131), (2, 0.039), (2, 0.431)],
)
def test_continuous_workspace_rejects_applied_pose_outside_box(axis, value) -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults(), continuous_workspace())
    current = np.array([0.70, 0.00, 0.20, 1, 0, 0, 0], dtype=float)
    current[axis] = value
    supervisor.latch_cycle_pose(current)
    requested = current.copy()

    with pytest.raises(SafetyFault, match="continuous workspace"):
        supervisor.limit_pose(requested, current)


def test_continuous_target_distance_is_measured_from_cycle_tcp() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults(), continuous_workspace())
    current = np.array([0.70, 0.00, 0.20, 1, 0, 0, 0], dtype=float)
    supervisor.latch_cycle_pose(current)
    requested = current.copy()
    requested[0] += 0.201

    with pytest.raises(SafetyFault, match="current TCP"):
        supervisor.limit_pose(requested, current)


def test_legacy_workspace_guard_remains_start_relative() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    start = np.array([0.70, 0.00, 0.20, 1, 0, 0, 0], dtype=float)
    supervisor.latch_start_pose(start)
    current = start.copy()
    current[0] += 0.079
    requested = start.copy()
    requested[0] += 0.082

    with pytest.raises(SafetyFault, match="workspace radius"):
        supervisor.limit_pose(requested, current)
