#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/90_formal/run_single_node_8gpu.sh <new-run-id>" >&2
  exit 2
fi

RUN_ID="$1"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SUITE="configs/suites/phase_b_f0.yaml"
LOCAL="configs/local/server.yaml"
SHARDS=8
OUTPUT_ROOT="$("$PYTHON" -c 'from task6_phaseb.common.config import load_config, root_for; print(root_for(load_config("configs/suites/phase_b_f0.yaml", "configs/local/server.yaml")))')"
LOG_ROOT="$OUTPUT_ROOT/tmp/formal-job-logs/$RUN_ID"
mkdir -p -- "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/orchestrator.log") 2>&1

on_exit() {
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "TASK6_PHASEB_F0_COMPLETE run_id=$RUN_ID"
  else
    echo "TASK6_PHASEB_F0_FAILED run_id=$RUN_ID exit_code=$status" >&2
  fi
}
trap on_exit EXIT

run_one() {
  label="$1"
  shift
  echo "===== START $label ====="
  "$@" 2>&1 | tee -a "$LOG_ROOT/$label.log"
  echo "===== DONE $label ====="
}

run_shards() {
  label="$1"
  launcher="$2"
  shift 2
  pids=()
  for ((shard=0; shard<SHARDS; shard++)); do
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="$shard" bash "$launcher" \
        --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID" \
        --shard-count "$SHARDS" --shard-index "$shard" "$@" \
        2>&1 | tee -a "$LOG_ROOT/$label-shard-$shard.log"
    ) &
    pids+=("$!")
  done
  failed=0
  for ((shard=0; shard<SHARDS; shard++)); do
    wait "${pids[$shard]}" || failed=1
  done
  [[ $failed -eq 0 ]] || return 1
}

count_complete() {
  root="$1"
  pattern="$2"
  find "$root" -type f -path "$pattern" | wc -l | tr -d '[:space:]'
}

[[ -f "$LOCAL" ]] || { echo "Create configs/local/server.yaml first" >&2; exit 1; }
[[ "$OUTPUT_ROOT" == /mnt/luoyulin_ckpt/* ]] || { echo "Unexpected output root: $OUTPUT_ROOT" >&2; exit 1; }
gpu_count="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
[[ "$gpu_count" == "$SHARDS" ]] || { echo "Expected 8 visible GPUs, found $gpu_count" >&2; exit 1; }

run_one preflight bash scripts/00_preflight/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one import-e64 bash scripts/55_import_e64/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one prepare env CUDA_VISIBLE_DEVICES=0 bash scripts/10_prepare/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one validate env CUDA_VISIBLE_DEVICES=0 bash scripts/20_validate/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_shards train scripts/30_train/run.sh

checkpoint_count="$(count_complete "$OUTPUT_ROOT/runs/train" "*/$RUN_ID/checkpoints/*/complete.json")"
[[ "$checkpoint_count" == 2640 ]] || { echo "Expected 2640 checkpoints, found $checkpoint_count" >&2; exit 1; }

run_shards capture-a scripts/40_capture/run.sh --part A
a_count="$(count_complete "$OUTPUT_ROOT/runs/capture/validation" "*/$RUN_ID/*/A/complete.json")"
[[ "$a_count" == 2656 ]] || { echo "Expected 2656 A captures, found $a_count" >&2; exit 1; }

run_one select-best bash scripts/40_capture/run.sh --part select-best --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_shards diagnostics scripts/40_capture/run.sh --part diagnostics
run_shards metrics scripts/50_metrics/run.sh --metric all
run_one aggregate bash scripts/60_aggregate/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one tables bash scripts/70_tables/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one figures bash scripts/80_figures/run.sh --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"

png_count="$(count_complete "$OUTPUT_ROOT/results/figures" "*/$RUN_ID/*.png")"
pdf_count="$(count_complete "$OUTPUT_ROOT/results/figures" "*/$RUN_ID/*.pdf")"
[[ "$png_count" -gt 0 && "$png_count" == "$pdf_count" ]] || {
  echo "Expected matching non-zero PNG/PDF figure counts; found png=$png_count pdf=$pdf_count" >&2
  exit 1
}
for experts in 128 256; do
  for section in main diagnostics appendix; do
    directory="$OUTPUT_ROOT/results/figures/${experts}_experts/$section/$RUN_ID"
    [[ -d "$directory" ]] || { echo "Missing required figure directory: $directory" >&2; exit 1; }
    [[ "$(find "$directory" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d '[:space:]')" -gt 0 ]] || {
      echo "No PNG figures in required directory: $directory" >&2
      exit 1
    }
  done
done
[[ ! -e "$OUTPUT_ROOT/results/figures/64_experts" ]] || {
  echo "E64 figures are out of scope for Phase B" >&2
  exit 1
}
[[ ! -e "$OUTPUT_ROOT/results/figures/expert_comparison" ]] || {
  echo "Cross-E figures are out of scope for Phase B" >&2
  exit 1
}
