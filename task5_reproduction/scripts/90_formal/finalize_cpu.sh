#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/90_formal/finalize_cpu.sh <existing-run-id>" >&2
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
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

MAIN_SUITE="configs/suites/main.yaml"
FORMAL_LOCAL="configs/local/formal.yaml"
METRIC_SHARDS="${METRIC_SHARDS:-8}"
OUTPUT_ROOT="$($PYTHON -c 'from task5.common.config import load_config, root_for; print(root_for(load_config("configs/suites/main.yaml", "configs/local/formal.yaml")))')"

if [[ "$OUTPUT_ROOT" != /mnt/luoyulin_ckpt/* ]]; then
  echo "CPU finalization output_root must be on /mnt/luoyulin_ckpt, found: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$OUTPUT_ROOT" || ! -w "$OUTPUT_ROOT" ]]; then
  echo "Output root is missing or not writable: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ "$(findmnt -n -o TARGET -T "$OUTPUT_ROOT")" != "/mnt/luoyulin_ckpt" ]]; then
  echo "Output root is not backed by the checkpoint mount: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is missing or not executable: $PYTHON" >&2
  exit 1
fi

LOG_ROOT="$OUTPUT_ROOT/tmp/formal-job-logs/$RUN_ID"
mkdir -p -- "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/cpu-finalize.log") 2>&1

on_exit() {
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "TASK5_CPU_FINALIZE_COMPLETE run_id=$RUN_ID"
  else
    echo "TASK5_CPU_FINALIZE_FAILED run_id=$RUN_ID exit_code=$status" >&2
  fi
}
trap on_exit EXIT

run_one() {
  local label="$1"
  shift
  echo "===== START $label ====="
  "$@" 2>&1 | tee -a "$LOG_ROOT/$label.log"
  echo "===== DONE $label ====="
}

count_files() {
  local root="$1"
  local pattern="$2"
  find "$root" -type f -path "$pattern" | wc -l | tr -d '[:space:]'
}

echo "TASK5 CPU-only finalizer"
echo "run_id=$RUN_ID"
echo "repo=$REPO_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "python=$PYTHON"
echo "metric_shards=$METRIC_SHARDS"
date -u +"utc=%Y-%m-%dT%H:%M:%SZ"

probe_count="$(count_files "$OUTPUT_ROOT/runs/capture/probe" "*/$RUN_ID/*/probe/complete.json")"
if [[ "$probe_count" != "2160" ]]; then
  echo "Expected 2160 complete probe captures, found $probe_count" >&2
  exit 1
fi
echo "Verified 2160 complete probe captures"

echo "===== START metric-coactivation_consistency: $METRIC_SHARDS CPU shards ====="
pids=()
for ((shard = 0; shard < METRIC_SHARDS; shard++)); do
  (
    set -o pipefail
    bash scripts/50_metrics/run.sh \
      --metric coactivation_consistency \
      --suite "$MAIN_SUITE" \
      --local "$FORMAL_LOCAL" \
      --run-id "$RUN_ID" \
      --shard-count "$METRIC_SHARDS" \
      --shard-index "$shard" \
      2>&1 | tee -a "$LOG_ROOT/metric-coactivation_consistency-shard-$shard.log"
  ) &
  pids+=("$!")
done

failed=0
for ((shard = 0; shard < METRIC_SHARDS; shard++)); do
  if wait "${pids[$shard]}"; then
    echo "coactivation metric shard $shard completed"
  else
    echo "coactivation metric shard $shard failed" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  exit 1
fi
echo "===== DONE metric-coactivation_consistency ====="

metric_count="$(count_files "$OUTPUT_ROOT/runs/metrics/coactivation_consistency" "*/$RUN_ID/*/metrics.json")"
if [[ "$metric_count" != "2160" ]]; then
  echo "Expected 2160 coactivation metric files, found $metric_count" >&2
  exit 1
fi
echo "Verified 2160 coactivation metric files"

run_one aggregate bash scripts/60_aggregate/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"
run_one tables bash scripts/70_tables/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"
run_one figures bash scripts/80_figures/run.sh \
  --suite "$MAIN_SUITE" --local "$FORMAL_LOCAL" --run-id "$RUN_ID"

png_count="$(count_files "$OUTPUT_ROOT/results/figures" "*/$RUN_ID/*.png")"
pdf_count="$(count_files "$OUTPUT_ROOT/results/figures" "*/$RUN_ID/*.pdf")"
table_count="$(count_files "$OUTPUT_ROOT/results/tables" "*/$RUN_ID/*")"
if [[ "$png_count" -eq 0 || "$png_count" != "$pdf_count" || "$table_count" -eq 0 ]]; then
  echo "Final artifact check failed: png=$png_count pdf=$pdf_count tables=$table_count" >&2
  exit 1
fi
echo "Final artifacts verified: png=$png_count pdf=$pdf_count tables=$table_count"
