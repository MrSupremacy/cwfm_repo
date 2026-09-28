#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <existing-routed-run-id-with-E64>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SUITE=configs/suites/phase_d_f0.yaml
LOCAL=configs/local/server.yaml
COMMON=(--suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID")
bash scripts/run.sh prepare-routed "${COMMON[@]}"
bash scripts/run.sh preflight-routed "${COMMON[@]}"
bash scripts/90_formal/run_router_8gpu.sh "$RUN_ID"

run_sharded() {
  local command="$1"
  shift
  local pids=()
  for shard in {0..7}; do
    (
      CUDA_VISIBLE_DEVICES="$shard" bash scripts/run.sh "$command" "${COMMON[@]}" \
        --shard-count 8 --shard-index "$shard" "$@"
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ $failed -eq 0 ]]
}

run_sharded capture --part A
bash scripts/run.sh capture "${COMMON[@]}" --part select-best
run_sharded capture --part diagnostics
run_sharded metrics --metric all
bash scripts/90_formal/finalize_router_cpu.sh "$RUN_ID"
echo "PHASE_D_ROUTED_COMPLETE run_id=$RUN_ID"
