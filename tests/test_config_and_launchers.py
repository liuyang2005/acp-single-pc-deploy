from __future__ import annotations

import subprocess
import os
from pathlib import Path

import pytest

from acp_single_pc_deploy.common.config import load_yaml_mapping
from acp_single_pc_deploy.robot.runner import (
    RunnerSettings,
    _make_continuous_workspace,
    _make_limits,
    build_arg_parser,
)


ROOT = Path(__file__).resolve().parents[1]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_fixed_inference_and_robot_configs() -> None:
    inference = load_yaml_mapping(ROOT / "configs" / "inference.yaml")
    robot = load_yaml_mapping(ROOT / "configs" / "robot.yaml")
    assert inference["network"]["bind_endpoint"] == "tcp://127.0.0.1:5555"
    assert inference["model"]["device"] == "cuda:0"
    assert inference["model"]["action_period_s"] == 0.5035
    assert inference["model"]["inference_seed"] == 42
    assert inference["model"]["expected_camera_view"] == "wrist"
    assert inference["model"]["minimum_checkpoint_epoch"] == 700
    assert robot["network"]["inference_endpoint"] == "tcp://127.0.0.1:5555"
    assert robot["robot"]["serial"] == "Rizon4s-063586"
    assert robot["robot"]["tool"] == "hapticexoteleop"
    assert robot["robot"]["home_joints_deg"] == [0, -32, 0, 90, 0, 28, 45]
    assert robot["camera"]["serial"] == "260322274925"
    assert robot["camera"]["dataset_name"] == "cam_260322274925_wrist"
    assert robot["camera"]["view"] == "wrist"
    assert robot["camera"]["color_order"] == "RGB"
    assert robot["camera"]["frame_timeout_ms"] == 15000
    assert robot["camera"]["start_attempts"] == 2
    assert robot["camera"]["retry_delay_s"] == 1.0
    assert robot["camera"]["start_timeout_s"] == 35.0
    assert robot["safety"]["stiffness_min_n_m"] == 200.0
    assert robot["safety"]["stiffness_max_n_m"] == 5000.0
    assert robot["safety"]["max_equivalent_target_radius_m"] == 0.20
    assert robot["safety"]["max_workspace_radius_m"] == 0.08
    assert (
        robot["safety"]["stiffness_max_n_m"]
        <= robot["execution"]["inner_translation_stiffness_n_m"]
    )
    assert robot["execution"]["execute_points"] == 4
    assert robot["execution"]["orientation_source"] == "reference"
    assert robot["continuous"]["execute_points"] == 2
    assert robot["continuous"]["commitment_points"] == 16
    assert robot["continuous"]["contact_force_threshold_n"] == 5.0
    assert robot["continuous"]["min_upward_exit_m"] == 0.03
    assert robot["acquisition"]["pose_sample_period_s"] == 0.01007
    assert robot["acquisition"]["wrench_sample_period_s"] == 0.005263
    assert robot["continuous"]["max_runtime_s"] == 120.0
    assert robot["continuous"]["workspace_min_xyz_m"] == [0.55, -0.14, 0.04]
    assert robot["continuous"]["workspace_max_xyz_m"] == [0.92, 0.13, 0.43]
    assert robot["continuous"]["max_equivalent_target_distance_m"] == 0.20
    assert not any(key.lower() == "zeroftsensor" or "zero_ft" in key.lower() for key in _all_keys(robot))


def test_policy_stiffness_cannot_exceed_inner_translation_stiffness() -> None:
    robot = load_yaml_mapping(ROOT / "configs" / "robot.yaml")
    robot["safety"]["stiffness_max_n_m"] = 5001.0

    with pytest.raises(ValueError, match="inner translation stiffness"):
        _make_limits(robot)


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


def test_runner_settings_reject_commitment_shorter_than_execution_window() -> None:
    with pytest.raises(ValueError, match="continuous_commitment_points"):
        RunnerSettings(
            continuous_execute_points=4,
            continuous_commitment_points=2,
        )


@pytest.mark.parametrize(
    "field",
    (
        "continuous_contact_force_threshold_n",
        "continuous_min_upward_exit_m",
    ),
)
def test_runner_settings_reject_nonpositive_commitment_thresholds(field) -> None:
    with pytest.raises(ValueError, match=field):
        RunnerSettings(**{field: 0.0})


def test_launcher_uses_fixed_checkpoint_and_conda_environments() -> None:
    inference_script = (ROOT / "run_inference.sh").read_text(encoding="utf-8")
    combined_script = (ROOT / "run_single_pc.sh").read_text(encoding="utf-8")
    assert "${1:?" in inference_script
    assert 'ACP_ENV="pyrite"' in combined_script
    assert 'ROBOT_ENV="haptic_exo_env"' in combined_script
    assert (
        'CHECKPOINT_PATH="${ACP_CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"'
        in combined_script
    )
    assert (
        'DEFAULT_CHECKPOINT_PATH="${HOME}/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints/2026.07.25_14.19.52_flip_up_new_conv_wrist_190hz_800ep/checkpoints/latest.ckpt"'
        in combined_script
    )
    assert (
        'MODE="${1:?usage: run_single_pc.sh '
        'dry-run|execute|continuous-dry-run|continuous}"'
        in combined_script
    )
    assert "${2:?" not in combined_script
    assert 'REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in combined_script
    assert "/acp_single_pc_deploy/run_" not in combined_script
    assert '[[ ! -f "${CHECKPOINT_PATH}" ]]' in combined_script
    assert "conda run" in combined_script
    assert "--health" in combined_script
    assert 'setsid conda run --no-capture-output -n "${ACP_ENV}"' in combined_script
    assert 'kill -- "-${INFERENCE_PID}"' in combined_script
    for mode in ("continuous_dry_run", "continuous"):
        assert (ROOT / f"run_{mode}.sh").is_file()
    assert "--mode continuous-dry-run" in (
        ROOT / "run_continuous_dry_run.sh"
    ).read_text(encoding="utf-8")
    assert "--mode continuous" in (ROOT / "run_continuous.sh").read_text(
        encoding="utf-8"
    )


def test_all_launchers_have_valid_bash_syntax() -> None:
    for path in ROOT.glob("run_*.sh"):
        bash_path = str(path)
        if os.name == "nt":
            relative = path.relative_to(path.anchor).as_posix()
            bash_path = f"/mnt/{path.drive[0].lower()}/{relative}"
        result = subprocess.run(["bash", "-n", bash_path], capture_output=True)
        error = result.stderr.decode("utf-8", errors="replace")
        assert result.returncode == 0, f"{path.name}: {error}"


def test_readme_documents_two_environment_hardware_gate() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "RTX 5060",
        "260322274925",
        "dry-run",
        "execute",
        "latest.ckpt",
        "紧急停止",
        "不调用 `ZeroFTSensor`",
        "continuous-dry-run",
        "continuous",
        "120",
        "0.55",
        "0.92",
        "Ctrl+C",
    ):
        assert required in text


def test_gitignore_excludes_runtime_and_local_machine_artifacts() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in (
        "__pycache__/",
        ".pytest_cache/",
        "logs/",
        "*.ckpt",
        ".env",
        ".venv/",
        "venv/",
    ):
        assert required in patterns


def test_shell_launchers_are_committed_with_lf_endings() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes.splitlines()
