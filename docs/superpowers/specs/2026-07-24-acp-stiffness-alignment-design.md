# ACP Stiffness Alignment Design

## Context

The first successful hardware dry-run decoded a valid ACP action chunk, but the
first predicted stiffness was `4941.82 N/m` and the deployment clipped it to
`1000 N/m`. This is a deployment mismatch: ACP labels use `200-5000 N/m`, and
the reference ACP executor uses the predicted value along the compliant axis
with `5000 N/m` on the two orthogonal translation axes.

Clipping the model output to `1000 N/m` changes the learned compliance response
and is not an acceptable way to execute the policy.

## Decision

1. Align the deployment stiffness range to `200-5000 N/m`.
2. Keep force, torque, velocity, acceleration, pose-step, and workspace limits
   unchanged.
3. Reject a configuration whose policy stiffness maximum exceeds the configured
   inner translation stiffness. The existing Flexiv connection check already
   rejects inner Cartesian stiffness above the robot-reported nominal stiffness,
   so these checks form a complete chain:

   `policy maximum <= inner translation stiffness <= robot nominal stiffness`.
4. Make dry-run simulate all configured execution points without sending robot
   commands. It must log each point and a summary.
5. Dry-run must fail if any point has a non-finite/invalid stiffness, stiffness
   clipping, or workspace fault. Translation and rotation step limiting remain
   active and are counted in the summary because those guards are expected to
   constrain a preview when necessary.
6. Workspace safety uses two distinct guards derived from the recorded task:
   the raw equivalent target must remain within `0.20 m` of the latched start,
   and the step-limited pose that could actually be sent must remain within
   `0.08 m`. The target guard runs before pose-step limiting; the applied-pose
   guard runs afterward.

## Alternatives Rejected

- Keep the `1000 N/m` cap: safe but materially changes the learned policy.
- Use an intermediate cap such as `2000` or `3000 N/m`: still changes the policy
  without a training-derived justification.
- Change only the YAML value: misses configuration consistency and full-chunk
  preview coverage.

## Runtime Flow

After inference returns one action chunk, dry-run reads the current robot pose,
latches it as the preview start, and evaluates all 12 configured points in time
order. Each simulated point uses the previous limited preview pose as its next
current pose. No call to `hardware.send_pose()` is allowed in dry-run.

The event log records predicted/applied stiffness, equivalent/applied pose, and
safety messages for every point, followed by a summary with point count and
limit counts. A successful dry-run transitions to `HOLD` with
`stop_reason=dry_run_complete` and `completed_command_steps=0`.

The two workspace radii are intentionally not interchangeable. The `0.20 m`
target radius rejects a runaway policy target, while the `0.08 m` applied radius
limits the actual command envelope for the guarded one-chunk deployment.

## Testing

- Configuration test asserts the `200-5000 N/m` policy range.
- Validation test rejects policy stiffness above inner translation stiffness.
- Runner test proves dry-run previews all configured points and sends no motion.
- Runner test proves stiffness clipping prevents dry-run success.
- Safety tests prove the raw equivalent target and final applied pose are checked
  against their respective workspace radii.
- Existing safety, hardware, inference, and launcher tests remain green.
- Hardware acceptance requires a new remote dry-run with no
  `stiffness_clipped`, return code zero, and zero command steps.
