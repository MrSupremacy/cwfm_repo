#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/90_formal/finalize_cpu.sh <existing-run-id>" >&2
  exit 2
fi
RUN_ID="$1"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
SUITE="configs/suites/phase_b_f0.yaml"
LOCAL="configs/local/server.yaml"
SHARDS="${METRIC_SHARDS:-8}"

pids=()
for ((shard=0; shard<SHARDS; shard++)); do
  bash scripts/50_metrics/run.sh --metric all --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID" \
    --shard-count "$SHARDS" --shard-index "$shard" &
  pids+=("$!")
done
failed=0
for ((shard=0; shard<SHARDS; shard++)); do
  wait "${pids[$shard]}" || failed=1
done
[[ $failed -eq 0 ]] || exit 1
bash scripts/60_aggregate/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
bash scripts/70_tables/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
bash scripts/80_figures/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
