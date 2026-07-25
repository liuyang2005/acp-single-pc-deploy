# test0724 deployment findings

The two continuous hardware runs used checkpoint SHA256
`4a5b52c9c4430c57c01bb2e11f938d5585c94da673e8f1fd6a4d37465204c44b`.
It matches `conv_wrist_190hz` at epoch 290, not the completed 800-epoch wrist
training run.

Both runs reported `action_period_s=0.15`. This came from multiplying action
stride 50 by the upstream runner's 2 ms raw period and a 1.5 slowdown. The
local dataset action timestamps instead come from the approximately 100 Hz
robot pose stream: the measured median period is 10.07 ms, so adjacent sparse
actions are approximately 0.5035 s apart.

The first prediction already contained a large lateral trajectory. In the
four-point run, the first reference path moved from y=-0.113 m to y=-0.052 m.
The twelve-point preview reached y=+0.016 m. The lateral motion therefore
existed in the policy output before equivalent-target conversion, while the
incorrect 0.15 s execution period made the robot chase later points much too
quickly.

The original online observation buffer sampled by array index at a 500 Hz poll
rate. That produced pose and wrench histories much shorter than training. It
also stopped appending robot states while a chunk was executing, so consecutive
inferences could combine old pre-chunk samples with one fresh sample.

The deployment now:

- rejects checkpoints that are not tagged as wrist-view or are below epoch 700;
- uses a fixed 0.5035 s action period and deterministic diffusion seed;
- samples all modalities by timestamp around the latest RGB frame;
- targets about 333 ms RGB, 101 ms pose, and 163 ms wrench histories;
- records robot state continuously inside the 200 Hz execution loop;
- rejects missing history instead of silently accepting a time gap;
- reobserves and replans every two continuous points, but preserves the active
  16-point plan until points 0 through 15 have been consumed; this prevents
  repeated execution of only the descending action prefix;
- executes four points in guarded single-chunk mode;
- uses reference orientation for the translation-only compliance labels;
- reports sample timestamps and history spans in each observation log.

The completed wrist checkpoint now selected by the launcher is
`conv_wrist_190hz_800ep` at epoch 790 with SHA256
`4f2ee74d3a8ca10fbd256f0be0b8d5517a932156d3fe60c18152e87d57aa277a`.
These changes remove known deployment-side confounders. A new dry-run with this
checkpoint is still required before attributing any remaining lateral prediction
to the learned policy or demonstrations.
