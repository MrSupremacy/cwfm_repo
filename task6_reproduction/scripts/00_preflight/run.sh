#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${1:-configs/local/server.yaml}"
RUN_ID="${2:-main01}"
df -h /mnt/luoyulin_ckpt/fanxuankai || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"${ROOT}/scripts/run.sh" preflight --local "${LOCAL}" --run-id "${RUN_ID}"
