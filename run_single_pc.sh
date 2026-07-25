#!/usr/bin/env bash
set -euo pipefail

# Operator settings: edit these three values when moving the deployment.
ACP_ENV="pyrite"
ROBOT_ENV="haptic_exo_env"
DEFAULT_CHECKPOINT_PATH="${HOME}/haptic_exo_teleop_ws/liuyang/Data/acp_checkpoints/2026.07.25_14.19.52_flip_up_new_conv_wrist_190hz_800ep/checkpoints/latest.ckpt"
CHECKPOINT_PATH="${ACP_CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"

MODE="${1:?usage: run_single_pc.sh dry-run|execute|continuous-dry-run|continuous}"

if [[ "${MODE}" != "dry-run" && "${MODE}" != "execute" \
    && "${MODE}" != "continuous-dry-run" && "${MODE}" != "continuous" ]]; then
  echo "mode must be dry-run, execute, continuous-dry-run, or continuous" >&2
  exit 2
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_SCRIPT="${REPO_DIR}/run_inference.sh"
ROBOT_SCRIPT="${REPO_DIR}/run_${MODE//-/_}.sh"
ENDPOINT="tcp://127.0.0.1:5555"
INFERENCE_PID=""

cleanup() {
  if [[ -n "${INFERENCE_PID}" ]] && kill -0 -- "-${INFERENCE_PID}" 2>/dev/null; then
    kill -- "-${INFERENCE_PID}" 2>/dev/null || true
    wait "${INFERENCE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_DIR}"
setsid conda run --no-capture-output -n "${ACP_ENV}" \
  bash "${INFERENCE_SCRIPT}" "${CHECKPOINT_PATH}" &
INFERENCE_PID=$!

ready=false
for _ in $(seq 1 60); do
  if ! kill -0 "${INFERENCE_PID}" 2>/dev/null; then
    echo "inference process exited before health check passed" >&2
    exit 1
  fi
  if conda run --no-capture-output -n "${ROBOT_ENV}" \
      python -m acp_single_pc_deploy.robot.client \
      --endpoint "${ENDPOINT}" --timeout 1.0 --health >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done

if [[ "${ready}" != "true" ]]; then
  echo "inference health check did not pass within 60 seconds" >&2
  exit 1
fi

conda run --no-capture-output -n "${ROBOT_ENV}" bash "${ROBOT_SCRIPT}"
