from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.robot.safety import (
    DeploymentState,
    SafetyFault,
    SafetyLimits,
    SafetySupervisor,
    WrenchReading,
)


def pose(x=0.0) -> np.ndarray:
    return np.array([x, 0, 0, 1, 0, 0, 0], dtype=np.float64)


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


def test_pose_is_step_limited_but_absolute_workspace_violation_faults() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    supervisor.latch_start_pose(pose())
    limited = supervisor.limit_pose(pose(0.01), current_pose7=pose())
    assert limited[0] == pytest.approx(0.002)
    assert "translation_step" in supervisor.last_limit_messages
    with pytest.raises(SafetyFault, match="workspace"):
        supervisor.limit_pose(pose(0.09), current_pose7=pose())


def test_stiffness_is_clipped_and_nonfinite_faults() -> None:
    supervisor = SafetySupervisor(SafetyLimits.defaults())
    assert supervisor.validate_stiffness(1500.0) == (1000.0, True)
    with pytest.raises(SafetyFault, match="stiffness"):
        supervisor.validate_stiffness(float("nan"))
