# ACP Continuous Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit `continuous-dry-run` and guarded 120-second `continuous` modes that synchronously infer fresh ACP actions and execute four points per request.

**Architecture:** Keep the inference protocol and one-chunk modes unchanged. Extend the safety supervisor with an optional continuous-workspace policy, then add a synchronous receding-horizon loop to `Runner` that owns request ordering, per-chunk execution, deadlines, events, and stop semantics. Expose the new behavior through validated YAML configuration and dedicated launchers.

**Tech Stack:** Python 3.10, NumPy, pytest, Flexiv RDK 1.9, Intel RealSense SDK, ZeroMQ, Bash, YAML.

---

## File Structure

- Modify `robot/safety.py`: define and enforce continuous absolute workspace and current-cycle target-distance limits without changing one-chunk limits.
- Modify `robot/runner.py`: validate continuous settings, run the synchronous policy loop, enforce deadlines, handle interrupts, and write continuous metadata/events.
- Modify `configs/robot.yaml`: add the approved four-point, 120-second, absolute XYZ, and 0.20 m continuous limits.
- Modify `tests/test_safety.py`: unit-test continuous pose guards and legacy guard isolation.
- Modify `tests/test_runner.py`: unit-test repeated inference, deadlines, failures, events, and mode compatibility.
- Modify `tests/conftest.py`: add deterministic fake-runtime controls required by multi-cycle tests.
- Modify `tests/test_config_and_launchers.py`: test configuration and launcher contracts.
- Create `run_continuous_dry_run.sh`: launch the robot process in non-commanding continuous mode.
- Create `run_continuous.sh`: launch the guarded real continuous mode.
- Modify `run_single_pc.sh`: accept and route all four modes while keeping the fixed checkpoint and Conda environments.
- Modify `README.md`: document the staged on-hardware acceptance procedure and exact commands.

### Task 1: Continuous Workspace Safety Policy

**Files:**
- Modify: `robot/safety.py`
- Test: `tests/test_safety.py`

- [ ] **Step 1: Write failing tests for absolute bounds and cycle-relative targets**

Append tests that construct an explicit continuous policy, latch the current
cycle pose, and verify both independent guards:

```python
from acp_single_pc_deploy.robot.safety import ContinuousWorkspaceLimits


def continuous_workspace() -> ContinuousWorkspaceLimits:
    return ContinuousWorkspaceLimits(
        minimum_xyz_m=np.array([0.55, -0.14, 0.04]),
        maximum_xyz_m=np.array([0.92, 0.13, 0.43]),
        max_equivalent_target_distance_m=0.20,
    )


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
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q tests/test_safety.py
```

Expected: failures because `ContinuousWorkspaceLimits` and
`latch_cycle_pose()` do not exist.

- [ ] **Step 3: Implement the continuous workspace value object and guard**

Add a frozen value object that copies and validates arrays in `__post_init__`:

```python
@dataclass(frozen=True)
class ContinuousWorkspaceLimits:
    minimum_xyz_m: np.ndarray
    maximum_xyz_m: np.ndarray
    max_equivalent_target_distance_m: float

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum_xyz_m, dtype=np.float64)
        maximum = np.asarray(self.maximum_xyz_m, dtype=np.float64)
        distance = float(self.max_equivalent_target_distance_m)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("continuous workspace bounds must have shape (3,)")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("continuous workspace bounds must be finite")
        if np.any(minimum >= maximum):
            raise ValueError("continuous workspace minimum must be below maximum")
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("continuous target distance must be finite and positive")
        object.__setattr__(self, "minimum_xyz_m", minimum.copy())
        object.__setattr__(self, "maximum_xyz_m", maximum.copy())
        object.__setattr__(self, "max_equivalent_target_distance_m", distance)
```

Extend `SafetySupervisor.__init__` with
`continuous_workspace: ContinuousWorkspaceLimits | None = None`, store it, and
add `_cycle_pose7`. Add:

```python
def latch_cycle_pose(self, pose7: np.ndarray) -> None:
    self._cycle_pose7 = self._validated_pose(pose7)
```

Replace `limit_pose` with the complete dual-policy implementation below. It
preserves the legacy start-relative branch when no continuous workspace is
configured. The continuous branch requires a latched cycle pose, rejects raw
equivalent targets beyond the configured distance, applies the same
translation/rotation step limits, then rejects the limited result outside any
inclusive XYZ bound:

```python
def limit_pose(self, requested_pose7: np.ndarray, current_pose7: np.ndarray) -> np.ndarray:
    requested = self._validated_pose(requested_pose7)
    current = self._validated_pose(current_pose7)
    continuous = self.continuous_workspace
    if continuous is not None:
        if self._cycle_pose7 is None:
            self.fault("continuous cycle pose is not latched")
        target_distance = float(np.linalg.norm(requested[:3] - self._cycle_pose7[:3]))
        if target_distance > continuous.max_equivalent_target_distance_m:
            self.fault(
                f"requested pose exceeds current TCP target distance: {target_distance:.6f}"
            )
    else:
        if self._start_pose7 is None:
            self.fault("start pose is not latched")
        target_radius = float(np.linalg.norm(requested[:3] - self._start_pose7[:3]))
        if target_radius > self.limits.max_equivalent_target_radius_m:
            self.fault(f"requested pose exceeds equivalent target radius: {target_radius:.6f}")

    result = requested.copy()
    messages: list[str] = []
    translation = requested[:3] - current[:3]
    distance = float(np.linalg.norm(translation))
    if distance > self.limits.max_translation_step_m:
        result[:3] = current[:3] + translation * (
            self.limits.max_translation_step_m / distance
        )
        messages.append("translation_step")
    dot = abs(float(np.dot(current[3:], requested[3:])))
    angle = 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))
    if angle > self.limits.max_rotation_step_rad:
        result[3:] = _slerp_quaternion(
            current[3:], requested[3:], self.limits.max_rotation_step_rad / angle
        )
        messages.append("rotation_step")

    if continuous is not None:
        xyz = result[:3]
        if np.any(xyz < continuous.minimum_xyz_m) or np.any(
            xyz > continuous.maximum_xyz_m
        ):
            self.fault(f"applied pose exceeds continuous workspace: {xyz.tolist()}")
    else:
        assert self._start_pose7 is not None
        applied_radius = float(np.linalg.norm(result[:3] - self._start_pose7[:3]))
        if applied_radius > self.limits.max_workspace_radius_m:
            self.fault(f"applied pose exceeds workspace radius: {applied_radius:.6f}")
    self._last_limit_messages = tuple(messages)
    return result
```

- [ ] **Step 4: Run focused and regression safety tests**

Run:

```bash
pytest -q tests/test_safety.py tests/test_executor.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the safety policy**

```bash
git add robot/safety.py tests/test_safety.py
git commit -m "Add continuous workspace safety policy"
```

### Task 2: Continuous Configuration and Mode Validation

**Files:**
- Modify: `configs/robot.yaml`
- Modify: `robot/runner.py`
- Modify: `tests/test_config_and_launchers.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing configuration and parser tests**

Add assertions to `test_fixed_inference_and_robot_configs`:

```python
assert robot["continuous"]["execute_points"] == 4
assert robot["continuous"]["max_runtime_s"] == 120.0
assert robot["continuous"]["workspace_min_xyz_m"] == [0.55, -0.14, 0.04]
assert robot["continuous"]["workspace_max_xyz_m"] == [0.92, 0.13, 0.43]
assert robot["continuous"]["max_equivalent_target_distance_m"] == 0.20
```

Add parser and validation tests:

```python
from acp_single_pc_deploy.robot.runner import (
    RunnerSettings,
    _make_continuous_workspace,
    build_arg_parser,
)


def test_parser_accepts_explicit_continuous_modes() -> None:
    parser = build_arg_parser()
    for mode in ("dry-run", "execute", "continuous-dry-run", "continuous"):
        args = parser.parse_args(["--mode", mode, "--config", "robot.yaml"])
        assert args.mode == mode


def test_continuous_config_rejects_inverted_workspace() -> None:
    robot = load_yaml_mapping(ROOT / "configs" / "robot.yaml")
    robot["continuous"]["workspace_min_xyz_m"][0] = 0.93
    with pytest.raises(ValueError, match="minimum"):
        _make_continuous_workspace(robot)


def test_runner_settings_reject_invalid_continuous_point_count() -> None:
    with pytest.raises(ValueError, match="continuous_execute_points"):
        RunnerSettings(continuous_execute_points=0)
```

- [ ] **Step 2: Run tests and verify missing continuous configuration fails**

Run:

```bash
pytest -q tests/test_config_and_launchers.py::test_fixed_inference_and_robot_configs \
  tests/test_config_and_launchers.py::test_parser_accepts_explicit_continuous_modes \
  tests/test_config_and_launchers.py::test_continuous_config_rejects_inverted_workspace \
  tests/test_config_and_launchers.py::test_runner_settings_reject_invalid_continuous_point_count
```

Expected: failures for missing config, parser choices, helper, and settings.

- [ ] **Step 3: Add and load the approved continuous configuration**

Add to `configs/robot.yaml`:

```yaml
continuous:
  execute_points: 4
  max_runtime_s: 120.0
  max_equivalent_target_distance_m: 0.20
  workspace_min_xyz_m: [0.55, -0.14, 0.04]
  workspace_max_xyz_m: [0.92, 0.13, 0.43]
```

Extend `RunnerSettings` and validate in `__post_init__`:

```python
continuous_execute_points: int = 4
max_continuous_runtime_s: float = 120.0

def __post_init__(self) -> None:
    if not 1 <= self.execute_points <= EXPECTED_CONTRACT.action_horizon:
        raise ValueError("execute_points must be within the action horizon")
    if not 1 <= self.continuous_execute_points <= EXPECTED_CONTRACT.action_horizon:
        raise ValueError("continuous_execute_points must be within the action horizon")
    if not np.isfinite(self.max_continuous_runtime_s) or self.max_continuous_runtime_s <= 0.0:
        raise ValueError("max_continuous_runtime_s must be finite and positive")
```

Add `_make_continuous_workspace(config)` to construct
`ContinuousWorkspaceLimits` from `config["continuous"]`. Require `continuous`
in the root config, pass its values into `RunnerSettings`, and allow all four
parser choices. Keep `Runner.__init__` validation aligned with the parser:

```python
MODES = ("dry-run", "execute", "continuous-dry-run", "continuous")

if mode not in MODES:
    raise ValueError(f"mode must be one of {', '.join(MODES)}")
```

Only pass `continuous_workspace` into `SafetySupervisor` for modes beginning
with `continuous`; legacy modes receive `None`. Add a
`continuous_workspace: ContinuousWorkspaceLimits | None` argument to `Runner`.
Reject construction when a continuous mode receives `None`, so real execution
cannot silently omit the absolute guard:

```python
is_continuous = mode in {"continuous-dry-run", "continuous"}
if is_continuous and continuous_workspace is None:
    raise ValueError("continuous modes require continuous workspace limits")
self.safety = SafetySupervisor(
    limits or SafetyLimits.defaults(),
    continuous_workspace if is_continuous else None,
)
```

In `main`, pass `_make_continuous_workspace(config)` for continuous modes. In
`Runner.for_test`, construct the approved bounds below for continuous modes and
pass `None` for legacy modes:

```python
ContinuousWorkspaceLimits(
    minimum_xyz_m=np.array([0.55, -0.14, 0.04]),
    maximum_xyz_m=np.array([0.92, 0.13, 0.43]),
    max_equivalent_target_distance_m=0.20,
)
```

- [ ] **Step 4: Run configuration and existing runner tests**

Run:

```bash
pytest -q tests/test_config_and_launchers.py tests/test_runner.py
```

Expected: all tests pass; new modes construct successfully but do not yet have
their final loop behavior.

- [ ] **Step 5: Commit configuration and mode contracts**

```bash
git add configs/robot.yaml robot/runner.py tests/test_config_and_launchers.py tests/test_runner.py
git commit -m "Configure ACP continuous deployment modes"
```

### Task 3: Synchronous Four-Point Receding-Horizon Loop

**Files:**
- Modify: `robot/runner.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Make the fake runtime deterministic across several chunks**

Extend `FakeComponents` with `observed_request_ids`, and record every ID:

```python
observed_request_ids: list[int] = field(default_factory=list)

def observe(self, request_id: int) -> ObservationPacket:
    self.observed_request_ids.append(request_id)
    packet = make_observation(request_id=request_id)
    packet.wrench[:] = self.hardware.raw_wrench
    return packet
```

Change `FakeHardware.pose7` to a valid pose inside the approved continuous
workspace so tests exercise policy behavior instead of failing on an invalid
fixture:

```python
self.pose7 = np.array([0.70, 0.00, 0.20, 1, 0, 0, 0], dtype=np.float64)
```

Allow `Runner.for_test` to accept test settings through an optional
`settings` argument, defaulting to the current fast settings. Tests will use a
small runtime rather than an unbounded fake loop.

- [ ] **Step 2: Write failing loop and dry-run tests**

Add:

```python
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


def test_continuous_executes_four_points_then_reobserves(fake_components) -> None:
    runner = Runner.for_test(
        "continuous", fake_components, settings=continuous_settings(max_continuous_runtime_s=0.04)
    )

    assert runner.run_once() == 0
    assert fake_components.observed_request_ids == list(
        range(len(fake_components.observed_request_ids))
    )
    assert len(fake_components.observed_request_ids) >= 2
    complete = [e for e in fake_components.events if e["type"] == "chunk_complete"]
    assert all(e["selected_point_count"] == 4 for e in complete)
    starts = [e for e in fake_components.events if e["type"] == "chunk_start"]
    selected = [e for e in fake_components.events if e["type"] == "action_selected_point"]
    assert len(selected) == 4 * len(starts)
    assert runner.completed_chunks == len(complete)


def test_continuous_dry_run_repeats_without_sending_pose(fake_components) -> None:
    runner = Runner.for_test(
        "continuous-dry-run", fake_components, settings=continuous_settings()
    )

    assert runner.run_once() == 0
    assert len(fake_components.observed_request_ids) >= 2
    assert fake_components.hardware.policy_pose_commands == []
    assert runner.stop_reason == "runtime_limit_reached"
```

- [ ] **Step 3: Run the two tests and verify they fail**

Run:

```bash
pytest -q tests/test_runner.py::test_continuous_executes_four_points_then_reobserves \
  tests/test_runner.py::test_continuous_dry_run_repeats_without_sending_pose
```

Expected: failures because the continuous loop and completed chunk counter do
not exist.

- [ ] **Step 4: Extract one-request inference and parameterize point handling**

Add `_infer_action(request_id)` that owns observation validation, inference,
and the existing complete `action_chunk` event. Refactor `_execute` and
`_preview` to accept `execute_points`; default callers still pass
`settings.execute_points`, continuous callers pass
`settings.continuous_execute_points`.

Before each continuous chunk, read a fresh robot state and call
`self.safety.latch_cycle_pose(state.pose7)`. For legacy one-chunk execution and
preview, continue to call `latch_start_pose`.

- [ ] **Step 5: Implement the synchronous continuous loop**

Add `completed_chunks = 0`, `_continuous_started_s`,
`_continuous_deadline_s`, and `_continuous_stop_emitted`. Establish the clock
and emit the start event with a helper:

```python
def _begin_continuous(self) -> None:
    self._continuous_started_s = self.clock()
    self._continuous_deadline_s = (
        self._continuous_started_s + self.settings.max_continuous_runtime_s
    )
    self._emit(
        "continuous_start",
        max_runtime_s=self.settings.max_continuous_runtime_s,
        execute_points=self.settings.continuous_execute_points,
    )
```

The loop receives the already validated first chunk, so the state transition
from `ARMED` to `RUNNING` is legal:

```python
def _run_continuous(self, first_chunk: Any) -> int:
    assert self._continuous_started_s is not None
    assert self._continuous_deadline_s is not None
    started_s = self._continuous_started_s
    deadline_s = self._continuous_deadline_s
    request_id = first_chunk.request_id
    chunk = first_chunk
    self._transition(DeploymentState.RUNNING, "continuous policy loop started")
    while self.clock() < deadline_s:
        self._emit(
            "chunk_start",
            request_id=request_id,
            chunk_index=self.completed_chunks,
            cumulative_runtime_s=self.clock() - started_s,
        )
        for point in range(self.settings.continuous_execute_points):
            self._emit(
                "action_selected_point",
                request_id=request_id,
                point=point,
                reference_pose7=chunk.reference_pose7[point],
                virtual_pose7=chunk.virtual_pose7[point],
                stiffness=chunk.stiffness[point],
            )
        if self.mode == "continuous-dry-run":
            completed = self._preview(
                chunk,
                execute_points=self.settings.continuous_execute_points,
                deadline_s=deadline_s,
            )
            command_count = 0
        else:
            before = self.completed_steps
            completed = self._execute(
                chunk,
                execute_points=self.settings.continuous_execute_points,
                deadline_s=deadline_s,
            )
            command_count = self.completed_steps - before
        if not completed:
            break
        self.completed_chunks += 1
        self._emit(
            "chunk_complete",
            request_id=request_id,
            chunk_index=self.completed_chunks - 1,
            selected_point_count=self.settings.continuous_execute_points,
            command_count=command_count,
            inference_latency_s=chunk.inference_latency_s,
            cumulative_runtime_s=self.clock() - started_s,
        )
        request_id += 1
        if self.clock() >= deadline_s:
            break
        chunk = self._infer_action(request_id)
    self.stop_reason = "runtime_limit_reached"
    self._emit(
        "continuous_stop",
        stop_reason=self.stop_reason,
        completed_chunks=self.completed_chunks,
        completed_command_steps=self.completed_steps,
        cumulative_runtime_s=self.clock() - started_s,
    )
    self._transition(DeploymentState.HOLD, self.stop_reason)
    return 0
```

`_execute` and `_preview` return `False` when `clock() >= deadline_s` before
the selected point window completes; otherwise they return `True`. Check the
deadline inside the 200 Hz `_execute` loop as well as before every preview
point.

Wire `run_once` with two explicit sequences. For `continuous-dry-run`, call
`_begin_continuous()` before `_infer_action(0)`, then transition to `ARMED` and
call `_run_continuous(first_chunk)`. For real `continuous`, infer request 0,
transition to `ARMED`, require the full serial, call `_begin_continuous()`
immediately before motion, and then call `_run_continuous(first_chunk)`. This
places the dry-run deadline before its first observation while keeping every
state transition valid. Keep the current one-chunk branches intact.

- [ ] **Step 6: Run runner and executor regression tests**

Run:

```bash
pytest -q tests/test_runner.py tests/test_executor.py
```

Expected: all tests pass, including legacy one-chunk expectations.

- [ ] **Step 7: Commit the synchronous loop**

```bash
git add robot/runner.py tests/conftest.py tests/test_runner.py
git commit -m "Add synchronous ACP continuous loop"
```

### Task 4: Stop Semantics, Failure Isolation, and Event Completeness

**Files:**
- Modify: `robot/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for interrupt, timeout, and partial chunks**

Add tests that inject failures without real hardware:

```python
def test_continuous_interrupt_is_normal_hold_and_cleans_up(fake_components) -> None:
    original_observe = fake_components.observe

    def interrupt_on_second_request(request_id):
        if request_id == 1:
            raise KeyboardInterrupt
        return original_observe(request_id)

    fake_components.observe = interrupt_on_second_request
    runner = Runner.for_test(
        "continuous", fake_components,
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
        "continuous", fake_components,
        settings=continuous_settings(max_continuous_runtime_s=0.1),
    )

    assert runner.run_once() == 1
    assert runner.stop_reason.startswith("inference_timeout:")
    assert [p.request_id for p in fake_components.client.inferred_packets] == [0]
    assert runner.completed_chunks == 1


def test_deadline_mid_chunk_does_not_increment_completed_chunks(fake_components) -> None:
    runner = Runner.for_test(
        "continuous", fake_components,
        settings=continuous_settings(max_continuous_runtime_s=0.006),
    )

    assert runner.run_once() == 0
    assert runner.completed_chunks == 0
    assert runner.completed_steps > 0
```

Import `InferenceTimeout` into the test module.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q tests/test_runner.py -k "interrupt or inference_timeout or deadline_mid_chunk"
```

Expected: failures for interrupt return/state, missing stop event, or incorrect
partial chunk accounting.

- [ ] **Step 3: Implement explicit normal-stop and fault-stop paths**

Add `_emit_continuous_stop()` and call it exactly once for every exit after
`continuous_start`. It computes elapsed time from `_continuous_started_s` and
returns without writing when `_continuous_stop_emitted` is already true. Catch
`KeyboardInterrupt` before `InferenceTimeout`:

```python
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
```

Make `_emit_continuous_stop` idempotent with a boolean flag. Unexpected errors
and `SafetyFault` keep the existing FAULT behavior, then emit the same final
continuous summary. Do not increment `completed_chunks` unless all four points
finish. Ensure `_execute` reads and validates robot/wrench state on every cycle
and never accesses an old chunk after `_infer_action` raises.

- [ ] **Step 4: Assert complete event fields**

Extend the successful-loop test to require:

```python
starts = [e for e in fake_components.events if e["type"] == "continuous_start"]
stops = [e for e in fake_components.events if e["type"] == "continuous_stop"]
assert len(starts) == len(stops) == 1
assert stops[0]["completed_chunks"] == runner.completed_chunks
assert stops[0]["completed_command_steps"] == runner.completed_steps
assert all("cumulative_runtime_s" in e for e in stops)
```

- [ ] **Step 5: Run all runner tests**

Run:

```bash
pytest -q tests/test_runner.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit stop and logging semantics**

```bash
git add robot/runner.py tests/test_runner.py
git commit -m "Harden ACP continuous stop semantics"
```

### Task 5: Runtime Artifacts and Metadata

**Files:**
- Modify: `robot/runner.py`
- Modify: `tests/test_config_and_launchers.py`
- Test: `tests/test_robot_runtime.py`

- [ ] **Step 1: Write failing tests for frame and metadata routing**

Extract the two pure helpers shown in Step 3 and test that frame capture applies to
`dry-run`, `continuous-dry-run`, and `continuous`, but not legacy `execute`:

```python
@pytest.mark.parametrize(
    "mode,expected",
    [
        ("dry-run", True),
        ("execute", False),
        ("continuous-dry-run", True),
        ("continuous", True),
    ],
)
def test_mode_saves_request_frames(mode, expected) -> None:
    assert should_save_request_frame(mode) is expected
```

Add a metadata test for `completed_chunks` and require the timing CSV header:

```python
assert timing_header() == [
    "request_id",
    "chunk_index",
    "inference_latency_s",
    "action_period_s",
    "selected_point_count",
    "command_count",
    "cumulative_runtime_s",
]
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
pytest -q tests/test_robot_runtime.py tests/test_config_and_launchers.py
```

Expected: failures because the pure routing/header helpers and new metadata do
not exist.

- [ ] **Step 3: Persist one frame and one timing row per request**

Add pure helpers:

```python
def should_save_request_frame(mode: str) -> bool:
    return mode in {"dry-run", "continuous-dry-run", "continuous"}


def timing_header() -> list[str]:
    return [
        "request_id", "chunk_index", "inference_latency_s", "action_period_s",
        "selected_point_count", "command_count", "cumulative_runtime_s",
    ]
```

Use `should_save_request_frame(args.mode)` in the observation closure. Write
timing rows from `chunk_complete`; retain a single-row one-chunk action record
for legacy modes using empty chunk-only fields. Add
`completed_chunks` to final metadata and `shutdown`.

- [ ] **Step 4: Run artifact and complete regression tests**

Run:

```bash
pytest -q tests/test_robot_runtime.py tests/test_config_and_launchers.py tests/test_runner.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit artifact support**

```bash
git add robot/runner.py tests/test_robot_runtime.py tests/test_config_and_launchers.py
git commit -m "Log ACP continuous run artifacts"
```

### Task 6: Dedicated Launchers and Operator Documentation

**Files:**
- Create: `run_continuous_dry_run.sh`
- Create: `run_continuous.sh`
- Modify: `run_single_pc.sh`
- Modify: `README.md`
- Modify: `tests/test_config_and_launchers.py`

- [ ] **Step 1: Write failing launcher contract tests**

Extend launcher tests:

```python
assert 'MODE="${1:?usage: run_single_pc.sh dry-run|execute|continuous-dry-run|continuous}"' in combined_script
for mode in ("continuous_dry_run", "continuous"):
    assert (ROOT / f"run_{mode}.sh").is_file()
assert "--mode continuous-dry-run" in (ROOT / "run_continuous_dry_run.sh").read_text()
assert "--mode continuous" in (ROOT / "run_continuous.sh").read_text()
```

Extend README required text with `continuous-dry-run`, `continuous`, `120`,
`0.55`, `0.92`, and `Ctrl+C`.

- [ ] **Step 2: Run launcher tests and verify they fail**

Run:

```bash
pytest -q tests/test_config_and_launchers.py
```

Expected: failures because the new scripts and documentation are absent.

- [ ] **Step 3: Create dedicated robot launchers**

Create both scripts using the established wrapper shape. The mode line is the
only semantic difference:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/acp_single_pc_deploy/configs/robot.yaml}"

cd "${ROOT_DIR}"
exec python -m acp_single_pc_deploy.robot.runner \
  --mode continuous-dry-run --config "${CONFIG}"
```

The real launcher uses `--mode continuous`. Update `run_single_pc.sh` to accept
the four exact modes and retain `${MODE//-/_}` routing, fixed
`ACP_ENV="pyrite"`, fixed `ROBOT_ENV="haptic_exo_env"`, and the fixed
checkpoint path.

- [ ] **Step 4: Document the mandatory staged hardware workflow**

Update README with exact commands:

```bash
bash acp_single_pc_deploy/run_single_pc.sh continuous-dry-run
bash acp_single_pc_deploy/run_single_pc.sh continuous
```

State that continuous dry-run homes the robot but sends no policy poses, both
continuous modes stop after 120 seconds, real mode executes four points per
request, `Ctrl+C` is a normal operator stop, and the physical emergency stop is
primary. Document the absolute XYZ bounds and 0.20 m current-TCP target guard.
Require on-site confirmation that the workspace is clear, emergency stop is
reachable, and gripper is already closed. State that real continuous execution
must not be launched automatically over SSH.

- [ ] **Step 5: Run launcher syntax and documentation tests**

Run:

```bash
pytest -q tests/test_config_and_launchers.py
bash -n run_single_pc.sh run_continuous_dry_run.sh run_continuous.sh
```

Expected: all tests pass and Bash reports no syntax errors.

- [ ] **Step 6: Commit launchers and runbook**

```bash
git add run_single_pc.sh run_continuous_dry_run.sh run_continuous.sh README.md tests/test_config_and_launchers.py
git commit -m "Add ACP continuous deployment launchers"
```

### Task 7: Full Verification and Remote Dry-Run Handoff

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run the complete local test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass with no skips newly introduced by this feature.

- [ ] **Step 2: Compile every Python module**

Run from the parent repository directory:

```bash
python -m compileall -q acp_single_pc_deploy
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Check patch hygiene and repository state**

Run:

```bash
git diff --check
git status --short
git log -8 --oneline
```

Expected: no whitespace errors; only intentional uncommitted changes, ideally
none; recent commits correspond to the tasks above.

- [ ] **Step 4: Push code without starting hardware**

Run:

```bash
git push origin main
```

Expected: GitHub `main` advances to the verified local HEAD.

- [ ] **Step 5: Update and run only continuous dry-run on the operation computer**

On `xense@192.168.1.149`, update the clean deployment checkout, then run:

```bash
cd ~/haptic_exo_teleop_ws/liuyang
git -C acp_single_pc_deploy pull --ff-only
bash acp_single_pc_deploy/run_single_pc.sh continuous-dry-run
```

Expected: after the homing confirmation, the program performs repeated requests
for at most 120 seconds, sends zero policy pose commands, exits with
`stop_reason: runtime_limit_reached`, and records sequential request IDs,
request PNG files, and no exception/fault events.

- [ ] **Step 6: Inspect the remote acceptance evidence**

Use the printed run directory:

```bash
python3 -m json.tool "$RUN_DIR/metadata.json"
grep -nE '"type":"(continuous_start|chunk_complete|continuous_stop|exception)"' \
  "$RUN_DIR/events.jsonl"
ls "$RUN_DIR/frames" | tail
```

Acceptance requires strictly increasing request IDs, four selected preview
points per completed chunk, no stiffness clipping, no stale sensor events, no
workspace prediction fault, and plausible inference latency. Do not proceed to
real motion if any acceptance condition fails.

- [ ] **Step 7: Stop for explicit real-hardware authorization**

Report the dry-run directory and evidence. Do not run
`run_single_pc.sh continuous` until the user explicitly confirms they are on
site, the workspace is clear, the emergency stop is reachable, and the gripper
is closed.
