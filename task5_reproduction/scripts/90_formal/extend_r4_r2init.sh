#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: SHARD_COUNT=<single-node-gpu-count> bash scripts/90_formal/extend_r4_r2init.sh formal20260830a" >&2
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

SUITE="configs/suites/main.yaml"
LOCAL="configs/local/formal_r4_r2init.yaml"
ARM="R4-R2Init"
OUTPUT_ROOT="$($PYTHON -c 'from task5.common.config import load_config, root_for; print(root_for(load_config("configs/suites/main.yaml", "configs/local/formal_r4_r2init.yaml")))')"
BASE_PROTOCOL="$($PYTHON -c 'from task5.common.config import load_config; print(load_config("configs/suites/main.yaml", "configs/local/formal_r4_r2init.yaml")["extension"]["base_protocol"])')"
VISIBLE_GPUS="$($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
SHARD_COUNT="${SHARD_COUNT:-$VISIBLE_GPUS}"

if [[ "$RUN_ID" != "formal20260830a" ]]; then
  echo "This extension configuration is bound to formal20260830a, found: $RUN_ID" >&2
  exit 1
fi
if [[ ! "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || (( SHARD_COUNT > VISIBLE_GPUS || SHARD_COUNT > 24 )); then
  echo "SHARD_COUNT must be between 1 and min(visible GPUs, 24); visible=$VISIBLE_GPUS requested=$SHARD_COUNT" >&2
  exit 1
fi
if [[ "$OUTPUT_ROOT" != /mnt/luoyulin_ckpt/* ]] || [[ ! -d "$OUTPUT_ROOT" || ! -w "$OUTPUT_ROOT" ]]; then
  echo "Invalid formal output root: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ "$(findmnt -n -o TARGET -T "$OUTPUT_ROOT")" != "/mnt/luoyulin_ckpt" ]]; then
  echo "Formal output root is not backed by /mnt/luoyulin_ckpt: $OUTPUT_ROOT" >&2
  exit 1
fi

# Fail before launching any GPU worker if the immutable base summaries are
# absent/mismatched or if this extension arm has already produced raw outputs.
for base_result in \
  "$OUTPUT_ROOT/results/data/normalized/$RUN_ID/metrics.json" \
  "$OUTPUT_ROOT/results/data/aggregated/$RUN_ID/metrics.json" \
  "$OUTPUT_ROOT/results/data/aggregated/$RUN_ID/paired_differences.json"
do
  if [[ ! -f "$base_result" ]]; then
    echo "Missing completed base result: $base_result" >&2
    exit 1
  fi
  "$PYTHON" -c \
    'import json, sys; meta=json.load(open(sys.argv[1], encoding="utf-8"))["meta"]; expected=(sys.argv[2], sys.argv[3], "main"); actual=(meta.get("protocol"), meta.get("run_id"), meta.get("suite")); actual == expected or sys.exit(f"Base result identity mismatch: {sys.argv[1]} actual={actual} expected={expected}")' \
    "$base_result" "$BASE_PROTOCOL" "$RUN_ID"
done

if find "$OUTPUT_ROOT/runs" -type d -path "*/$ARM/default/*/*/$RUN_ID" -print -quit | grep -q .; then
  echo "Refusing to mix with existing $ARM raw outputs for run $RUN_ID" >&2
  exit 1
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "R4-R2Init extension launch checks passed: run_id=$RUN_ID shards=$SHARD_COUNT output_root=$OUTPUT_ROOT"
  exit 0
fi

LOG_ROOT="$OUTPUT_ROOT/tmp/formal-job-logs/$RUN_ID"
mkdir -p -- "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/extend-r4-r2init.log") 2>&1

on_exit() {
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "TASK5_R4_R2INIT_EXTENSION_COMPLETE run_id=$RUN_ID"
  else
    echo "TASK5_R4_R2INIT_EXTENSION_FAILED run_id=$RUN_ID exit_code=$status" >&2
  fi
}
trap on_exit EXIT

run_one() {
  local label="$1"
  shift
  echo "===== START $label ====="
  "$@"
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
      CUDA_VISIBLE_DEVICES="$shard" bash "$launcher" \
        --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID" \
        --arm "$ARM" --shard-count "$SHARD_COUNT" --shard-index "$shard" \
        "${extra[@]}"
    ) &
    pids+=("$!")
  done
  local failed=0
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    wait "${pids[$shard]}" || failed=1
  done
  (( failed == 0 )) || return 1
  echo "===== DONE $label ====="
}

count_files() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -path "$pattern" | wc -l | tr -d '[:space:]'
}

run_one preflight bash scripts/00_preflight/run.sh \
  --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID" --arm "$ARM"

run_shards train scripts/30_train/run.sh

checkpoints="$(count_files "$OUTPUT_ROOT/runs/train" "*/$ARM/default/*/*/$RUN_ID/checkpoints/*/complete.json")"
[[ "$checkpoints" == "264" ]] || { echo "Expected 264 R4-R2Init checkpoints, found $checkpoints" >&2; exit 1; }

run_shards capture-a scripts/40_capture/run.sh --part A --skip-complete

captures_a="$(count_files "$OUTPUT_ROOT/runs/capture/validation" "*/$ARM/default/*/*/$RUN_ID/*/A/complete.json")"
[[ "$captures_a" == "264" ]] || { echo "Expected 264 R4-R2Init A captures, found $captures_a" >&2; exit 1; }

run_one select-best bash scripts/40_capture/run.sh \
  --part select-best --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID" --arm "$ARM"

run_shards diagnostics scripts/40_capture/run.sh --part diagnostics --skip-complete

probes="$(count_files "$OUTPUT_ROOT/runs/capture/probe" "*/$ARM/default/*/*/$RUN_ID/*/probe/complete.json")"
loads="$(count_files "$OUTPUT_ROOT/runs/capture/validation" "*/$ARM/default/*/*/$RUN_ID/*/B/complete.json")"
[[ "$probes" == "264" ]] || { echo "Expected 264 R4-R2Init probe captures, found $probes" >&2; exit 1; }
(( loads >= 24 && loads <= 48 )) || { echo "Expected 24..48 deduplicated R4-R2Init B captures, found $loads" >&2; exit 1; }

for metric in performance load_balance churn oracle_overlap activation_coverage coactivation_consistency; do
  run_shards "metric-$metric" scripts/50_metrics/run.sh --metric "$metric"
done

# aggregate snapshots the old summaries under results/data/extension_base before
# it writes the combined view. Tables and figures may then replace old renderings.
run_one aggregate-extension bash scripts/60_aggregate/run.sh \
  --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one tables bash scripts/70_tables/run.sh \
  --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"
run_one figures bash scripts/80_figures/run.sh \
  --suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID"

echo "R4-R2Init extension complete: 24 runs, 264 checkpoints/A/probe states; base outputs preserved in results/data/extension_base/$RUN_ID"
