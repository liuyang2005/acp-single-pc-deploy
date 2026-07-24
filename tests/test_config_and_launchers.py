from __future__ import annotations

import subprocess
import os
from pathlib import Path

from acp_single_pc_deploy.common.config import load_yaml_mapping


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
    assert inference["model"]["raw_time_step_s"] == 0.002
    assert inference["model"]["slow_down_factor"] == 1.5
    assert robot["network"]["inference_endpoint"] == "tcp://127.0.0.1:5555"
    assert robot["robot"]["serial"] == "Rizon4s-063586"
    assert robot["robot"]["tool"] == "hapticexoteleop"
    assert robot["robot"]["home_joints_deg"] == [0, -32, 0, 90, 0, 28, 45]
    assert robot["camera"]["serial"] == "260322274925"
    assert robot["camera"]["dataset_name"] == "cam_260322274925_wrist"
    assert robot["camera"]["color_order"] == "RGB"
    assert robot["camera"]["frame_timeout_ms"] == 15000
    assert robot["camera"]["start_attempts"] == 2
    assert robot["camera"]["retry_delay_s"] == 1.0
    assert robot["camera"]["start_timeout_s"] == 35.0
    assert robot["execution"]["execute_points"] == 12
    assert not any(key.lower() == "zeroftsensor" or "zero_ft" in key.lower() for key in _all_keys(robot))


def test_launcher_uses_fixed_checkpoint_and_conda_environments() -> None:
    inference_script = (ROOT / "run_inference.sh").read_text(encoding="utf-8")
    combined_script = (ROOT / "run_single_pc.sh").read_text(encoding="utf-8")
    assert "${1:?" in inference_script
    assert 'ACP_ENV="pyrite"' in combined_script
    assert 'ROBOT_ENV="haptic_exo_env"' in combined_script
    assert (
        'CHECKPOINT_PATH="${HOME}/haptic_exo_teleop_ws/liuyang/acp_checkpoints/latest.ckpt"'
        in combined_script
    )
    assert 'MODE="${1:?' in combined_script
    assert "${2:?" not in combined_script
    assert '[[ ! -f "${CHECKPOINT_PATH}" ]]' in combined_script
    assert "conda run" in combined_script
    assert "--health" in combined_script


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
