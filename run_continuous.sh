#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${REPO_DIR}/configs/robot.yaml}"

cd "${REPO_DIR}"
exec python -m acp_single_pc_deploy.robot.runner \
  --mode continuous --config "${CONFIG}"
