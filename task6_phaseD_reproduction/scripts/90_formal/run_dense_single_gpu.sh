#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <new-dense-run-id>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
COMMON=(--suite configs/suites/dense_mt.yaml --local configs/local/server.yaml --run-id "$RUN_ID")
bash scripts/run.sh preflight-dense "${COMMON[@]}"
bash scripts/run.sh train-dense "${COMMON[@]}"
bash scripts/run.sh evaluate-dense "${COMMON[@]}"
bash scripts/run.sh select-dense-best "${COMMON[@]}"
bash scripts/run.sh dense-results "${COMMON[@]}"
echo "DENSE_MT_REVIEW_READY run_id=$RUN_ID"

