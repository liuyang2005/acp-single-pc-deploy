# ACP Public Repository Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the deployment environment names and publish `acp_single_pc_deploy` as an independent public GitHub repository.

**Architecture:** Keep the deployment directory self-contained as the Git root. Test fixed launcher settings first, add repository exclusions, verify all software checks, inspect the exact staged scope, then create and push the new public GitHub repository.

**Tech Stack:** Bash, Python, pytest, Git, GitHub CLI.

---

### Task 1: Operation-Computer Environment Names

**Files:**
- Modify: `tests/test_config_and_launchers.py`
- Modify: `run_single_pc.sh`
- Modify: `README.md`

- [ ] **Step 1: Change the launcher test to require the new names**

```python
assert 'ACP_ENV="pyrite"' in combined_script
assert 'ROBOT_ENV="haptic_exo_env"' in combined_script
```

- [ ] **Step 2: Run the focused test and verify RED**

Run `pytest -q tests/test_config_and_launchers.py::test_launcher_uses_fixed_checkpoint_and_conda_environments` from the deployment directory with its parent on `PYTHONPATH`.

Expected: FAIL because the script still contains `acp_deploy` and `data_collect`.

- [ ] **Step 3: Update script and README**

Set `ACP_ENV="pyrite"` and `ROBOT_ENV="haptic_exo_env"` in `run_single_pc.sh`, and use the same values in the README example.

- [ ] **Step 4: Run the focused launcher tests**

Run `pytest -q tests/test_config_and_launchers.py`. Expected: all tests pass.

### Task 2: Repository Exclusions And Verification

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Add ignore-policy assertions**

Extend the launcher/config test file to assert `.gitignore` includes `__pycache__/`, `.pytest_cache/`, `logs/`, `*.ckpt`, `.env`, and common virtual-environment directories.

- [ ] **Step 2: Run the ignore-policy test and verify RED**

Expected: FAIL because `.gitignore` does not exist.

- [ ] **Step 3: Add the root `.gitignore`**

Exclude Python caches, test caches, coverage output, virtual environments, editor metadata, runtime logs/frames, checkpoints, environment files, and OS metadata.

- [ ] **Step 4: Run complete local verification**

Run all deployment tests, `python -m compileall -q .`, and `bash -n` for all four launchers. Expected: all pass.

### Task 3: Initialize And Publish Repository

**Files:**
- Create: `.git/` through `git init`

- [ ] **Step 1: Initialize the independent repository**

Run `git init -b main` from `acp_single_pc_deploy` and verify its top-level path equals that directory.

- [ ] **Step 2: Stage and inspect the exact scope**

Run `git add .`, inspect `git status --short`, `git diff --cached --stat`, and `git ls-files`. Require no cache, log, checkpoint, `.env`, or parent-directory path.

- [ ] **Step 3: Create the initial commit**

Commit with message `Initial ACP single-PC deployment`.

- [ ] **Step 4: Create and push the public GitHub repository**

Run `gh repo create liuyang2005/acp-single-pc-deploy --public --source . --remote origin --push`.

- [ ] **Step 5: Verify publication**

Verify the origin URL, `main` upstream, clean worktree, local/remote commit SHA match, repository visibility is `PUBLIC`, and default branch is `main`.
