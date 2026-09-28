#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/90_formal/resume_single_node_8gpu.sh <existing-run-id>" >&2
  exit 2
fi

RUN_ID="$1"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$REPO_ROOT"

export PYTHON="${PYTHON:-/opt/task5-venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MAIN_SUITE="configs/suites/main.yaml"
FORMAL_LOCAL="configs/local/formal.yaml"
SHARD_COUNT=8
PLAN_SCRIPT="scripts/90_formal/recovery_plan.py"
OUTPUT_ROOT="$($PYTHON -c 'from task5.common.config import load_config, root_for; print(root_for(load_config("configs/suites/main.yaml", "configs/local/formal.yaml")))')"

if [[ "$OUTPUT_ROOT" != /mnt/luoyulin_ckpt/* ]]; then
  echo "Recovery output_root must be on /mnt/luoyulin_ckpt, found: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$OUTPUT_ROOT" || ! -w "$OUTPUT_ROOT" ]]; then
  echo "Recovery output_root is missing or not writable: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ "$(findmnt -n -o TARGET -T "$OUTPUT_ROOT")" != "/mnt/luoyulin_ckpt" ]]; then
  echo "Recovery output_root is not backed by the checkpoint mount: $OUTPUT_ROOT" >&2
  exit 1
fi

LOG_ROOT="$OUTPUT_ROOT/tmp/formal-job-logs/$RUN_ID"
mkdir -p -- "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/recovery-orchestrator.log") 2>&1

on_exit() {
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "TASK5_FORMAL_RECOVERY_COMPLETE run_id=$RUN_ID"
  else
    echo "TASK5_FORMAL_RECOVERY_FAILED run_id=$RUN_ID exit_code=$status" >&2
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

build_recovery_plans() {
  local shard
  echo "===== START build and verify recovery plans ====="
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    "$PYTHON" "$PLAN_SCRIPT" \
      --suite "$MAIN_SUITE" \
      --local "$FORMAL_LOCAL" \
      --run-id "$RUN_ID" \
      --shard-count "$SHARD_COUNT" \
      --shard-index "$shard" \
      --verify-hashes \
      > "$LOG_ROOT/recovery-plan-shard-$shard.tsv"
  done
  echo "===== DONE build and verify recovery plans ====="
}

run_recovery_shard() {
  local shard="$1"
  local plan="$LOG_ROOT/recovery-plan-shard-$shard.tsv"
  local action task arm variant k seed checkpoint
  while IFS=$'\t' read -r action task arm variant k seed checkpoint; do
    [[ -n "$action" ]] || continue
    local -a args=(
      --suite "$MAIN_SUITE"
      --local "$FORMAL_LOCAL"
      --run-id "$RUN_ID"
      --task "$task"
      --arm "$arm"
      --variant "$variant"
      --k "$k"
      --seed "$seed"
    )
    if [[ "$action" == "resume" ]]; then
      args+=(--resume "$checkpoint")
    elif [[ "$action" != "start" ]]; then
      echo "Unknown recovery action: $action" >&2
      return 1
    fi
    echo "RECOVERY action=$action task=$task arm=$arm variant=$variant k=$k seed=$seed checkpoint=$checkpoint"
    CUDA_VISIBLE_DEVICES="$shard" bash scripts/30_train/run.sh "${args[@]}"
  done < "$plan"
}

run_recovery_shards() {
  local -a pids=()
  local shard
  echo "===== START recovery-train: $SHARD_COUNT independent single-GPU shards ====="
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    (
      set -o pipefail
      run_recovery_shard "$shard" 2>&1 | tee -a "$LOG_ROOT/recovery-train-shard-$shard.log"
    ) &
    pids+=("$!")
  done
  local failed=0
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    if wait "${pids[$shard]}"; then
      echo "recovery-train shard $shard completed"
    else
      echo "recovery-train shard $shard failed" >&2
      failed=1
    fi
  done
  if [[ $failed -ne 0 ]]; then
    echo "recovery-train failed; later stages will not run" >&2
    return 1
  fi
  echo "===== DONE recovery-train ====="
}

echo "TASK5 formal single-node recovery launcher"
echo "run_id=$RUN_ID"
echo "repo=$REPO_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "python=$PYTHON"
date -u +"utc=%Y-%m-%dT%H:%M:%SZ"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is missing or not executable: $PYTHON" >&2
  exit 1
fi
gpu_count="$($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$gpu_count" != "$SHARD_COUNT" ]]; then
  echo "Expected $SHARD_COUNT visible GPUs, found $gpu_count" >&2
  exit 1
fi
nvidia-smi -L
"$PYTHON" -m pip check
df -hT "$OUTPUT_ROOT"

run_one recovery-preflight "" bash scripts/00_preflight/run.sh \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

build_recovery_plans
run_recovery_shards

"$PYTHON" "$PLAN_SCRIPT" \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID" \
  --require-complete \
  > /dev/null

checkpoint_count="$(count_files "$OUTPUT_ROOT/runs/train" "*/$RUN_ID/checkpoints/*/complete.json")"
if [[ "$checkpoint_count" != "2112" ]]; then
  echo "Expected 2112 complete checkpoints, found $checkpoint_count" >&2
  exit 1
fi
echo "Verified 2112 complete checkpoints"

run_shards capture-a scripts/40_capture/run.sh --part A --skip-complete

capture_a_count="$(count_files "$OUTPUT_ROOT/runs/capture/validation" "*/$RUN_ID/*/A/complete.json")"
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

run_shards diagnostics scripts/40_capture/run.sh --part diagnostics --skip-complete

run_one capture-e-sst2 0 bash scripts/40_capture/run.sh \
  --part E --task sst2 --skip-complete \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

run_one capture-e-mnli 0 bash scripts/40_capture/run.sh \
  --part E --task mnli --skip-complete \
  --suite "$MAIN_SUITE" \
  --local "$FORMAL_LOCAL" \
  --run-id "$RUN_ID"

coactivation_count="$(count_files "$OUTPUT_ROOT/artifacts/coactivation" "*/$RUN_ID/complete.json")"
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

png_count="$(count_files "$OUTPUT_ROOT/results/figures" "*/$RUN_ID/*.png")"
pdf_count="$(count_files "$OUTPUT_ROOT/results/figures" "*/$RUN_ID/*.pdf")"
table_count="$(count_files "$OUTPUT_ROOT/results/tables" "*/$RUN_ID/*")"
if [[ "$png_count" -eq 0 || "$png_count" != "$pdf_count" || "$table_count" -eq 0 ]]; then
  echo "Final artifact check failed: png=$png_count pdf=$pdf_count tables=$table_count" >&2
  exit 1
fi

echo "Final artifacts verified: png=$png_count pdf=$pdf_count tables=$table_count"
