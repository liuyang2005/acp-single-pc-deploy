# ACP Stiffness Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align deployed ACP stiffness with its `200-5000 N/m` training range while preserving hardware and motion safety gates and validating every point in dry-run.

**Architecture:** Keep stiffness decoding in `ActionChunkExecutor` unchanged. Validate the configuration chain when constructing safety limits, rely on the existing Flexiv nominal-stiffness check for the hardware boundary, and expand `Runner._preview()` into a sequential 12-point simulation that never calls `send_pose()`.

**Tech Stack:** Python 3.10, NumPy, pytest, YAML configuration, Flexiv RDK 1.9, JSONL event logging.

---

### Task 1: Align And Validate The Stiffness Configuration

**Files:**
- Modify: `configs/robot.yaml:53-54`
- Modify: `robot/runner.py:450-470`
- Test: `tests/test_config_and_launchers.py`

- [ ] **Step 1: Write failing configuration tests**

Update `test_fixed_inference_and_robot_configs()` to assert:

```python
assert robot["safety"]["stiffness_min_n_m"] == 200.0
assert robot["safety"]["stiffness_max_n_m"] == 5000.0
assert robot["safety"]["stiffness_max_n_m"] <= robot["execution"]["inner_translation_stiffness_n_m"]
```

Add a test that loads the config, sets `stiffness_max_n_m` above
`inner_translation_stiffness_n_m`, and asserts `_make_limits()` raises:

```python
def test_policy_stiffness_cannot_exceed_inner_translation_stiffness() -> None:
    robot = load_yaml_mapping(ROOT / "configs" / "robot.yaml")
    robot["safety"]["stiffness_max_n_m"] = 5001.0
    with pytest.raises(ValueError, match="inner translation stiffness"):
        _make_limits(robot)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest -q acp_single_pc_deploy/tests/test_config_and_launchers.py
```

Expected: failure because the config still contains `1000.0` and `_make_limits()`
does not reject a policy maximum above inner translation stiffness.

- [ ] **Step 3: Implement minimal configuration alignment and validation**

Set `stiffness_max_n_m: 5000.0` in `configs/robot.yaml`. In `_make_limits()`,
parse the policy bounds and inner translation stiffness before constructing
`SafetyLimits`, then reject inconsistent values:

```python
stiffness_min = float(safety["stiffness_min_n_m"])
stiffness_max = float(safety["stiffness_max_n_m"])
inner_stiffness = float(execution["inner_translation_stiffness_n_m"])
if not 0.0 < stiffness_min <= stiffness_max <= inner_stiffness:
    raise ValueError(
        "policy stiffness range must be positive, ordered, and not exceed inner translation stiffness"
    )
```

Pass `stiffness_min` and `stiffness_max` into `SafetyLimits`. Do not change force,
torque, speed, step, or workspace values.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command. Expected: all tests in the file pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add configs/robot.yaml robot/runner.py tests/test_config_and_launchers.py
git commit -m "Align ACP policy stiffness range"
```

### Task 2: Preview Every Executed Point In Dry-Run

**Files:**
- Modify: `robot/runner.py:320-342`
- Modify: `tests/conftest.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing full-preview tests**

Give `FakeComponents` an `events` list and pass `event_sink=components.events.append`
from `Runner.for_test()`. Add a test proving all configured points are previewed:

```python
def test_dry_run_previews_every_execution_point(fake_components) -> None:
    runner = Runner.for_test("dry-run", fake_components)
    assert runner.run_once() == 0
    points = [event for event in fake_components.events if event["type"] == "action_preview_point"]
    assert len(points) == runner.settings.execute_points == 12
    assert [event["point"] for event in points] == list(range(12))
    assert fake_components.hardware.policy_pose_commands == []
```

Add a test proving clipping fails dry-run:

```python
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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest -q acp_single_pc_deploy/tests/test_runner.py
```

Expected: no `action_preview_point` events exist and stiffness clipping does not
currently fail dry-run.

- [ ] **Step 3: Implement sequential full-chunk preview**

In `_preview()`, retain one real `read_state()` call, latch the real start pose,
and create `ActionChunkExecutor` as today. Iterate exactly `execute_points` times.
For point `i`, evaluate at `start_time + i * action_period_s`, pass the previous
limited preview pose as `current_pose7`, and emit:

```python
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
```

If `stiffness_clipped` is present, call `self.safety.fault()` with the point index.
Count `translation_step` and `rotation_step`, then emit one `action_preview`
summary containing `point_count`, both limit counts, and `stiffness_clip_count=0`.
Never call `hardware.send_pose()` in `_preview()`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same runner pytest command. Expected: all runner tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add robot/runner.py tests/conftest.py tests/test_runner.py
git commit -m "Preview full ACP action chunk in dry-run"
```

### Task 3: Document, Verify, Push, And Re-run Hardware Dry-Run

**Files:**
- Modify: `README.md:86-93`
- Verify: all repository Python and shell files

- [ ] **Step 1: Update the operator acceptance criteria**

Document that the ACP policy range is `200-5000 N/m`, the existing Flexiv
nominal check protects the hardware boundary, and dry-run must report 12 preview
points with zero stiffness clips before execute.

- [ ] **Step 2: Run complete local verification**

Run:

```bash
python -m pytest -q acp_single_pc_deploy/tests
python -m compileall -q acp_single_pc_deploy
git -C acp_single_pc_deploy diff --check
```

Expected: all tests pass, compileall exits zero, and diff-check prints nothing.
The launcher syntax tests included in pytest must also pass.

- [ ] **Step 3: Commit documentation and push all commits**

```bash
git add README.md
git commit -m "Document ACP stiffness dry-run gate"
git push origin main
```

- [ ] **Step 4: Pull and run remote hardware dry-run**

On `192.168.1.149`, fast-forward the deployment repo, verify the existing
inference service health, then run only the robot launcher in `haptic_exo_env`:

```bash
git -C ~/haptic_exo_teleop_ws/liuyang/acp_single_pc_deploy pull --ff-only origin main
conda run --no-capture-output -n haptic_exo_env \
  bash ~/haptic_exo_teleop_ws/liuyang/acp_single_pc_deploy/run_dry_run.sh
```

- [ ] **Step 5: Verify remote acceptance evidence**

Inspect the new `metadata.json` and `events.jsonl`. Acceptance requires all of:

```text
return_code = 0
stop_reason = dry_run_complete
completed_command_steps = 0
action_preview_point count = 12
stiffness_clip_count = 0
no exception or startup_exception event
no remaining robot runner process
```

Do not run `execute` as part of this plan.

## Approved Safety Revision

Hardware dry-run showed that the original single `0.08 m` check ran against the
raw equivalent target before step limiting. The approved correction adds
`max_equivalent_target_radius_m: 0.20` for that raw target and retains
`max_workspace_radius_m: 0.08` for the step-limited pose that could be sent.
Both guards require focused safety tests and remain part of the remote dry-run
acceptance gate.
