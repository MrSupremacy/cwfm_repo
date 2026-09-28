#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <existing-routed-run-id-with-E64>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SUITE=configs/suites/phase_d_f0.yaml
LOCAL=configs/local/server.yaml
pids=()
for shard in {0..7}; do
  (
    CUDA_VISIBLE_DEVICES="$shard" bash scripts/run.sh train-router \
      --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID" \
      --shard-count 8 --shard-index "$shard"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]]
