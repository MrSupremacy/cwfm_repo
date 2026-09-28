#!/usr/bin/env bash
# Single-node independent runs, NOT eight-GPU DDP for a single model.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHON="${PYTHON:-/opt/task5-venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUN_ID="${1:-formal20260830a}"
STAGE="${2:-all}"
[[ $# -le 2 && "$RUN_ID" == "formal20260830a" ]] || { echo "Usage: bash $0 formal20260830a [all|train|capture|metrics|report|check]" >&2; exit 2; }
case "$STAGE" in all|train|capture|metrics|report|check) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
COMMON=(--suite configs/suites/phaseA_f0.yaml --local configs/extensions/phaseA_f0.yaml --run-id "$RUN_ID")

# Always validate old source/config/input identities before any expensive work.
"$PYTHON" -m task5.cli phase-a-check "${COMMON[@]}"
if [[ "$STAGE" == check || "${CHECK_ONLY:-0}" == 1 ]]; then
  "$PYTHON" -m task5.cli matrix "${COMMON[@]}"
  exit 0
fi

SHARD_COUNT="${SHARD_COUNT:-8}"
[[ "$SHARD_COUNT" =~ ^[1-8]$ ]] || { echo "SHARD_COUNT must be 1..8 (default 8)" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
if [[ "$STAGE" == all || "$STAGE" == train || "$STAGE" == capture ]]; then
  (( ${#GPU_IDS[@]} >= SHARD_COUNT )) || { echo "Not enough GPU_IDS" >&2; exit 2; }
  # Resolve masks before spawning; respect user-supplied GPU IDs/UUIDs.
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")" "$PYTHON" -c \
    'import sys, torch; n=int(sys.argv[1]); torch.cuda.device_count() >= n or sys.exit("Not enough visible GPUs"); print("Independent GPU workers:", n)' "$SHARD_COUNT"
  declare -A seen=()
  for ((i=0; i<SHARD_COUNT; i++)); do
    [[ -n "${GPU_IDS[$i]}" && -z "${seen[${GPU_IDS[$i]}]:-}" ]] || { echo "Empty/duplicate GPU ID" >&2; exit 2; }
    seen[${GPU_IDS[$i]}]=1
  done
fi

run_shards() {
  local command="$1" shard failed=0
  shift
  local -a pids=()
  for ((shard=0; shard<SHARD_COUNT; shard++)); do
    echo "$command: worker $shard/$SHARD_COUNT (independent conditions)"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]:-}" "$PYTHON" -m task5.cli "$command" "${COMMON[@]}" \
      --shard-count "$SHARD_COUNT" --shard-index "$shard" "$@" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "Stage failed: $command; no next stage launched" >&2; return 1; }
}

if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  # Trainable filter is applied BEFORE sharding: 24 R4-hard runs, 3/GPU at 8 GPUs.
  # Refuses existing train directories; never overwrites/resets checkpoints.
  run_shards train --arm R4-hard
fi
if [[ "$STAGE" == all || "$STAGE" == capture ]]; then
  # Both new arms only: 264 trained states + 8 static states.
  run_shards capture --part A --skip-complete
  "$PYTHON" -m task5.cli capture "${COMMON[@]}" --part select-best
  run_shards capture --part diagnostics --skip-complete
fi
if [[ "$STAGE" == all || "$STAGE" == metrics ]]; then
  run_shards metrics --metric all
fi
if [[ "$STAGE" == all || "$STAGE" == report ]]; then
  "$PYTHON" -m task5.cli phase-a-report "${COMMON[@]}"
fi
echo "Phase A F0 stage completed: $STAGE. Old arms/results were not recomputed or overwritten."
