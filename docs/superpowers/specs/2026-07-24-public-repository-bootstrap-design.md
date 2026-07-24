# ACP Public Repository Bootstrap Design

Date: 2026-07-24

## Goal

Publish `acp_single_pc_deploy` as an independent public GitHub repository after
changing its fixed Conda environment names to match the operation computer.

## Runtime Settings

`run_single_pc.sh` and the Chinese README will use:

```bash
ACP_ENV="pyrite"
ROBOT_ENV="haptic_exo_env"
```

The fixed checkpoint path remains:

```bash
CHECKPOINT_PATH="${HOME}/haptic_exo_teleop_ws/liuyang/acp_checkpoints/latest.ckpt"
```

Tests will assert all three settings and the simplified mode-only launcher
interface.

## Repository Boundary

The Git root will be `acp_single_pc_deploy`, not its parent
`haptic-exo-teleop`. The initial commit will contain deployment source,
configuration, launchers, requirements, tests, README, and design documents.

A root `.gitignore` will exclude Python bytecode and caches, pytest caches,
virtual environments, editor metadata, runtime logs, generated dry-run frames,
and local checkpoint files. No collected data, checkpoint, secret, or parent
repository content will be staged.

## GitHub Publication

The repository will be created under the authenticated account as:

```text
https://github.com/liuyang2005/acp-single-pc-deploy
```

It will be public, use `main` as the default branch, and use `origin` for the
remote. After tests and staged-file inspection pass, one initial commit will be
pushed directly to `main`. A pull request is unnecessary because this is a new
repository with no pre-existing default branch.

## Verification

Before publication:

- run the complete `acp_single_pc_deploy/tests` suite;
- compile all Python modules;
- validate all Bash launchers with `bash -n`;
- inspect `git status`, staged paths, and staged diff statistics;
- verify no cache, log, checkpoint, or secret file is tracked.

After publication, verify `origin`, upstream tracking, clean worktree, remote
visibility, default branch, and the pushed commit SHA.
