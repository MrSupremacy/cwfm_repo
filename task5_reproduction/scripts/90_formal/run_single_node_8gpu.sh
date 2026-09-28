#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/90_formal/run_single_node_8gpu.sh <new-run-id>" >&2
  exit 2
fi

RUN_ID="$1"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$REPO_ROOT"

export PYTHON="${PYTHON:-/opt/task5-venv/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MAIN_SUITE="configs/suites/main.yaml"
FORMAL_LOCAL="configs/local/formal.yaml"
SHARD_COUNT=8
LOG_ROOT="tmp/formal-job-logs/$RUN_ID"
mkdir -p -- "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/orchestrator.log") 2>&1

on_exit() {
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "TASK5_FORMAL_COMPLETE run_id=$RUN_ID"
  else
    echo "TASK5_FORMAL_FAILED run_id=$RUN_ID exit_code=$status" >&2
  fi
}
trap on_exit EXIT

run_one() {
  local label="$1"
  local visible_devices="$2"
  shift 2
  echo "===== START $label ====="
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$visible_devices" "$@" 2>&1 | tee -a "$LOG_ROOT/$label.log"
  )
  echo "===== DONE $label ====="
}

run_shards() {
  local label="$1"
  local launcher="$2"
  shift 2
  local -a extra=("$@")
  local -a pids=()
  local shard

  echo "===== START $label: $SHARD_COUNT independent single-GPU shards ====="
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="$shard" bash "$launcher" \
        --suite "$MAIN_SUITE" \
        --local "$FORMAL_LOCAL" \
        --run-id "$RUN_ID" \
        --shard-count "$SHARD_COUNT" \
        --shard-index "$shard" \
        "${extra[@]}" \
        2>&1 | tee -a "$LOG_ROOT/$label-shard-$shard.log"
    ) &
    pids+=("$!")
  done

  local failed=0
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    if wait "${pids[$shard]}"; then
      echo "$label shard $shard completed"
    else
      echo "$label shard $shard failed" >&2
      failed=1
    fi
  done
  if [[ $failed -ne 0 ]]; then
    echo "$label failed; later stages will not run" >&2
    return 1
  fi
  echo "===== DONE $label ====="
}

count_files() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -path "$pattern" | wc -l | tr -d '[:space:]'
}

echo "TASK5 formal single-node launcher"
echo "run_id=$RUN_ID"
echo "repo=$REPO_ROOT"
echo "python=$PYTHON"
date -u +"utc=%Y-%m-%dT%H:%M:%SZ"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is missing or not executable: $PYTHON" >&2
  exit 1
fi

gpu_count="$($PYTHON -c "import torch; print(torch.cuda.device_count())")"
if [[ "$gpu_count" != "$SHARD_COUNT" ]]; then
  echo "Expected $SHARD_COUNT visible GPUs, found $gpu_count" >&2
  exit 1
fi
nvidia-smi -L
"$PYTHON" -m pip check

run_one preflight "" bash scripts/00_preflight/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

# Formal preparation and Phase 0 are protocol stages, not a repeat of smoke.
run_one prepare 0 bash scripts/10_prepare/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

run_one phase0 0 bash scripts/20_validate/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

run_shards train scripts/30_train/run.sh

checkpoint_count="$(count_files runs/train "*/$RUN_ID/checkpoints/*/complete.json")"
if [[ "$checkpoint_count" != "2112" ]]; then
  echo "Expected 2112 complete checkpoints, found $checkpoint_count" >&2
  exit 1
fi
echo "Verified 2112 complete checkpoints"

run_shards capture-a scripts/40_capture/run.sh --part A

capture_a_count="$(count_files runs/capture/validation "*/$RUN_ID/*/A/complete.json")"
if [[ "$capture_a_count" != "2162" ]]; then
  echo "Expected 2162 complete Capture A states, found $capture_a_count" >&2
  exit 1
fi
echo "Verified 2162 complete Capture A states"

run_one select-best "" bash scripts/40_capture/run.sh \
  --part select-best \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

run_shards diagnostics scripts/40_capture/run.sh --part diagnostics

# Shared E is task-level output and must never be sharded by condition.
run_one capture-e-sst2 0 bash scripts/40_capture/run.sh \
  --part E --task sst2 \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

run_one capture-e-mnli 0 bash scripts/40_capture/run.sh \
  --part E --task mnli \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

coactivation_count="$(count_files artifacts/coactivation "*/$RUN_ID/complete.json")"
if [[ "$coactivation_count" != "2" ]]; then
  echo "Expected 2 complete task-level coactivation outputs, found $coactivation_count" >&2
  exit 1
fi
echo "Verified task-level coactivation outputs"

for metric in \
  performance \
  load_balance \
  churn \
  oracle_overlap \
  activation_coverage \
  coactivation_consistency
do
  run_one "metric-$metric" "" bash scripts/50_metrics/run.sh \
    --metric "$metric" \
    --suite "$MAIN_SUITE" \
    --local "$FORMAL_LOCAL" \
    --run-id "$RUN_ID"
done

run_one aggregate "" bash scripts/60_aggregate/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

run_one tables "" bash scripts/70_tables/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

run_one figures "" bash scripts/80_figures/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

png_count="$(count_files results/figures "*/$RUN_ID/*.png")"
pdf_count="$(count_files results/figures "*/$RUN_ID/*.pdf")"
table_count="$(count_files results/tables "*/$RUN_ID/*")"
if [[ "$png_count" -eq 0 || "$png_count" != "$pdf_count" || "$table_count" -eq 0 ]]; then
  echo "Final artifact check failed: png=$png_count pdf=$pdf_count tables=$table_count" >&2
  exit 1
fi

echo "Final artifacts verified: png=$png_count pdf=$pdf_count tables=$table_count"
