#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <run-id>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash scripts/90_formal/require_8gpu.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/90_formal/run_p0.sh "$RUN_ID"
bash scripts/90_formal/run_p1_full_8gpu.sh "$RUN_ID"
bash scripts/90_formal/run_p1_replay_8gpu.sh "$RUN_ID"
bash scripts/30_results/run.sh "$RUN_ID"
echo "TASK8_P01_COMPLETE run_id=$RUN_ID"
