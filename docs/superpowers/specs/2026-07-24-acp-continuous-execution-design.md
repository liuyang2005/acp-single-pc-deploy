# ACP Continuous Execution Design

Date: 2026-07-24

## Objective

Add a guarded continuous deployment mode for the ACP policy on the single
operation computer. The controller repeatedly captures a fresh observation,
runs inference, and executes the first four points of the returned action
chunk. Existing `dry-run` and one-chunk `execute` modes remain unchanged.

The continuous mode is intended for the existing Flexiv robot, D405 wrist
camera `260322274925`, and ACP checkpoint contract. It does not control the
gripper.

## Selected Architecture

Use a synchronous receding-horizon loop:

1. Capture a fresh synchronized observation.
2. Send it to inference with a strictly increasing request ID.
3. Validate the matching action response.
4. Execute the first four action points.
5. Repeat from a new observation until a stop condition occurs.

This keeps observation-to-action ownership explicit and prevents an old action
from being reused after an inference or sensor failure. Inference and motion do
not overlap in the first continuous implementation. At the ACP action period of
approximately 0.15 seconds, four points produce a new policy update roughly
every 0.6 seconds, plus observation and inference latency.

## Modes and Compatibility

The launcher and runner gain two explicit modes:

- `continuous-dry-run`: performs startup and repeated observation/inference
  cycles without sending policy pose commands.
- `continuous`: performs startup once and then runs the synchronous closed-loop
  controller.

The existing modes retain their current contracts:

- `dry-run` previews one full action chunk without policy motion.
- `execute` executes one action chunk and stops.

Continuous support must not silently change the behavior or safety thresholds
of these existing modes.

## Startup and State Flow

Startup is shared with the existing runner:

1. Connect to the robot and require workspace-clear homing confirmation.
2. Move to the established ACP/FOAR initial joint configuration.
3. Start the wrist camera and wait for a usable frame.
4. Sample the wrist-wrench baseline.
5. Complete the inference handshake and validate the model contract.
6. Capture and validate the first observation and action.
7. For `continuous`, require the existing full robot-serial confirmation before
   any policy command is sent.

After confirmation, the state moves from `ARMED` to `RUNNING`. A normal runtime
deadline or operator interrupt moves the controller to `HOLD`, followed by
cleanup and robot `IDLE`. A safety or unexpected runtime error moves it to
`FAULT`, followed by the same best-effort cleanup.

## Continuous Control Loop

Request IDs start at zero and increase by exactly one per inference cycle. A
response is usable only when its request ID matches the current observation.

For each request:

1. Capture a fresh RGB, pose, and wrench history.
2. Validate sensor ages and current raw and baseline-relative wrench limits.
3. Run inference and validate the action schema and request ID.
4. Log the complete predicted action chunk.
5. In `continuous-dry-run`, validate and preview the four selected points but
   send no robot commands.
6. In `continuous`, execute only points 0 through 3 using the existing 200 Hz
   control loop, pose step limiting, stiffness handling, and per-cycle robot and
   wrench checks.
7. Log chunk completion and begin a new request from a fresh observation.

No prior action is retained as a fallback. If observation or inference fails,
the controller stops instead of continuing the previous trajectory.

For `continuous`, the 120-second deadline begins immediately before the first
policy motion. For `continuous-dry-run`, the same deadline begins immediately
before its first observation/inference cycle. It is checked during action
execution as well as between chunks, so a real-motion chunk cannot extend
execution materially beyond the deadline and a dry-run cannot loop forever.

## Safety Model

All existing per-cycle checks remain active:

- robot fault and operational state;
- raw force and torque norms;
- baseline-relative force and torque norms;
- predicted stiffness validity and configured range;
- maximum translation and rotation per control step;
- configured robot velocity, acceleration, stiffness, and contact-wrench
  limits;
- RGB, pose, and wrench freshness at every new observation.

Continuous mode replaces the start-relative applied workspace-radius guard with
an absolute applied TCP translation box:

| Axis | Minimum | Maximum |
| --- | ---: | ---: |
| x | 0.55 m | 0.92 m |
| y | -0.14 m | 0.13 m |
| z | 0.04 m | 0.43 m |

Every pose that could be sent to the robot must remain inside this box. The
raw equivalent target for each action evaluation must also be no farther than
0.20 m from the robot TCP measured at the beginning of that inference/execution
cycle. This current-TCP target guard replaces the one-chunk guard's distance
from the initially latched pose only in continuous modes.

The physical emergency stop remains the primary emergency control. Software
stops include:

- `runtime_limit_reached`: normal stop at 120 seconds;
- `operator_interrupt`: normal stop after `Ctrl+C`;
- `inference_timeout`: stop without reusing an action;
- sensor, robot, force, torque, stiffness, target-distance, or workspace
  violations: immediate fault;
- request ID mismatch or unexpected runtime exception: immediate fault.

Cleanup always attempts to close the camera and inference client, stop robot
policy control, and return the robot to `IDLE`. Cleanup errors are logged and do
not prevent remaining resources from being stopped.

## Configuration

Continuous-only settings are explicit configuration values rather than changes
to one-chunk defaults:

- executed points per inference: `4`;
- maximum continuous run duration: `120.0 s`;
- absolute TCP bounds: the x/y/z limits listed above;
- equivalent-target distance from current cycle TCP: `0.20 m`.

The configured four-point count must be positive and no greater than the action
horizon returned by the inference handshake. The continuous workspace bounds
must be finite and ordered. Existing policy stiffness limits remain 200 to 5000
N/m.

## Events and Artifacts

Each run keeps its own existing run directory. Continuous runs add these event
types:

- `continuous_start`: mode, deadline, point count, and workspace limits;
- `chunk_start`: request ID, chunk index, and cumulative motion time;
- `chunk_complete`: request ID, chunk index, points selected, command count,
  inference latency, and cumulative motion time;
- `continuous_stop`: final stop reason, completed chunks, completed command
  steps, and cumulative motion time.

Existing observation, action-chunk, preview, command, exception, transition,
and shutdown events remain available. Each request logs:

- request ID and latest observation timing;
- inference latency and full predicted action chunk;
- the first four selected action points;
- per-command raw and delta wrench, predicted and applied stiffness, equivalent
  target, applied pose, and safety-limit messages;
- completed command count and cumulative runtime.

The latest wrist RGB frame for each request is stored using the request ID in
its filename. Shutdown metadata retains the global `completed_command_steps`
and adds the number of completed chunks. Every exit has an explicit
`stop_reason`.

## Error Handling

`KeyboardInterrupt` is handled separately from unexpected exceptions so it is
recorded as `operator_interrupt` and returns through the normal hold/cleanup
path. Deadline expiry is also a successful, normal stop. Inference timeout
stops in `HOLD`; invalid or unsafe data and unexpected exceptions stop in
`FAULT`.

A chunk counts as complete only after all four selected points have completed.
If a deadline, interrupt, or fault occurs mid-chunk, executed command steps are
still logged, but the completed-chunk counter is not incremented.

## Verification

Automated tests must cover:

- multiple synchronous cycles and exactly four selected points per request;
- strictly increasing request IDs and rejection of mismatched IDs;
- fresh observations for every inference and no stale-action fallback;
- 120-second expiry between and during chunks;
- `Ctrl+C`, inference timeout, sensor failure, and robot failure cleanup;
- absolute workspace acceptance and rejection on every axis;
- current-TCP equivalent-target distance acceptance and rejection;
- correct event fields and completed chunk/command counters;
- `continuous-dry-run` never sending a policy pose;
- regression behavior of existing `dry-run` and one-chunk `execute` modes.

Before real continuous motion:

1. Run all local tests and Python compilation checks.
2. Run `continuous-dry-run` on the 5060 operation computer.
3. Inspect request ordering, inference latency, saved wrist frames, stiffness,
   target-distance checks, and predicted workspace positions.
4. Obtain a new explicit on-site confirmation that the workspace is clear, the
   emergency stop is reachable, and the gripper is already closed.
5. Start the real `continuous` launcher locally on the operation computer.

Real continuous robot execution must not be started automatically over SSH as
part of implementation or verification.

## Out of Scope

- asynchronous inference and execution;
- action prefetching or overlapping chunks;
- policy-provided task completion detection;
- gripper control;
- changing the trained ACP model or checkpoint;
- changing existing one-chunk safety behavior.
