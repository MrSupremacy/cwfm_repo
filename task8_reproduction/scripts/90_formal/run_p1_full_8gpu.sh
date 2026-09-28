#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <run-id>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
bash scripts/90_formal/require_8gpu.sh
pids=()
for shard in {0..7}; do
  (
    CUDA_VISIBLE_DEVICES="$shard" bash scripts/20_p1/run_full.sh "$shard" 8 "$RUN_ID"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]]
echo "TASK8_P1_FULL_COMPLETE run_id=$RUN_ID"
