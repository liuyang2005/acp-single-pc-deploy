#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/acp_single_pc_deploy/configs/robot.yaml}"

cd "${ROOT_DIR}"
exec python -m acp_single_pc_deploy.robot.runner \
  --mode continuous --config "${CONFIG}"
