#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${1:?usage: run_inference.sh /absolute/path/to/latest.ckpt}"
CONFIG="${2:-${REPO_DIR}/configs/inference.yaml}"

cd "${REPO_DIR}"
exec python -m acp_single_pc_deploy.inference.server \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}"
