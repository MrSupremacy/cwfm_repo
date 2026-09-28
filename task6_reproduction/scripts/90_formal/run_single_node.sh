#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${1:-configs/local/server.yaml}"
RUN_ID="${2:-main01}"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
  echo "Formal Phase A launcher requires exactly 8 visible GPUs; found ${GPU_COUNT}" >&2
  exit 1
fi
LOG_ROOT="${ROOT}/tmp/formal_launch/${RUN_ID}"
mkdir -p "${LOG_ROOT}"
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${gpu}" "${ROOT}/scripts/run.sh" train --local "${LOCAL}" --run-id "${RUN_ID}" \
    --shard-index "${gpu}" --shard-count 8 >"${LOG_ROOT}/gpu_${gpu}.log" 2>&1 &
done
wait
