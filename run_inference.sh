#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?usage: run_inference.sh /absolute/path/to/latest.ckpt}"
CONFIG="${2:-${ROOT_DIR}/acp_single_pc_deploy/configs/inference.yaml}"

cd "${ROOT_DIR}"
exec python -m acp_single_pc_deploy.inference.server \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}"
