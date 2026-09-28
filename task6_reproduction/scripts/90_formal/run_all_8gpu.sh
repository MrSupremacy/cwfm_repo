#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${1:-configs/local/server.yaml}"
RUN_ID="${2:-}"
F0_RESULTS="${3:-}"

if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 LOCAL_CONFIG NEW_RUN_ID [TASK5_F0_NORMALIZED_JSON]" >&2
  exit 2
fi

cd "${ROOT}"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
  echo "Formal Phase A requires exactly 8 visible GPUs; found ${GPU_COUNT}" >&2
  exit 1
fi

LOG_ROOT="${ROOT}/tmp/formal_launch/${RUN_ID}"
mkdir -p "${LOG_ROOT}"

run_8_shards() {
  local stage="$1"
  shift
  local pids=()
  local status=0
  for gpu in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES="${gpu}" "${ROOT}/scripts/run.sh" "$@" \
      --local "${LOCAL}" --run-id "${RUN_ID}" \
      --shard-index "${gpu}" --shard-count 8 \
      >"${LOG_ROOT}/${stage}_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "Stage ${stage} failed; inspect ${LOG_ROOT}/${stage}_gpu_*.log" >&2
    return 1
  fi
}

echo "[1/11] dependency and unit/integration tests"
"${ROOT}/scripts/20_validate/test_local.sh" >"${LOG_ROOT}/tests.log" 2>&1

echo "[2/11] formal matrix and real-input preflight"
"${ROOT}/scripts/run.sh" preflight --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/preflight.log" 2>&1

echo "[3/11] prepare fixed probes and initial centroids"
CUDA_VISIBLE_DEVICES=0 "${ROOT}/scripts/run.sh" prepare --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/prepare.log" 2>&1

echo "[4/11] Phase 0 real-model validation"
CUDA_VISIBLE_DEVICES=0 "${ROOT}/scripts/run.sh" validate --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/validate.log" 2>&1

echo "[5/11] 198 fullFT training runs (192 routed + 6 dense) on eight independent shards"
run_8_shards train train

echo "[6/11] capture A for every checkpoint and dense reference"
run_8_shards capture_A capture --part A

echo "[7/11] select each run's best validation checkpoint"
CUDA_VISIBLE_DEVICES="" "${ROOT}/scripts/run.sh" capture --part select-best \
  --local "${LOCAL}" --run-id "${RUN_ID}" >"${LOG_ROOT}/select_best.log" 2>&1

echo "[8/11] capture best/final B and all-checkpoint C+D"
run_8_shards capture_diagnostics capture --part diagnostics

echo "[9/11] compute five metric groups on eight condition shards"
run_8_shards metrics metrics

echo "[10/11] aggregate, validate completeness, and render tables/figures"
CUDA_VISIBLE_DEVICES="" "${ROOT}/scripts/run.sh" aggregate --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/aggregate.log" 2>&1
CUDA_VISIBLE_DEVICES="" "${ROOT}/scripts/run.sh" phase-a-check --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/phase_a_check.log" 2>&1
CUDA_VISIBLE_DEVICES="" "${ROOT}/scripts/run.sh" tables --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/tables.log" 2>&1
CUDA_VISIBLE_DEVICES="" "${ROOT}/scripts/run.sh" figures --local "${LOCAL}" --run-id "${RUN_ID}" \
  >"${LOG_ROOT}/figures.log" 2>&1

echo "[11/11] optional Task 5 F0 comparison"
if [[ -n "${F0_RESULTS}" ]]; then
  CUDA_VISIBLE_DEVICES="" "${ROOT}/scripts/run.sh" compare-f0 --local "${LOCAL}" \
    --run-id "${RUN_ID}" --f0-results "${F0_RESULTS}" >"${LOG_ROOT}/compare_f0.log" 2>&1
else
  echo "No Task 5 F0 normalized JSON supplied; F1 workflow is complete and F0 comparison was skipped."
fi

echo "Task 6 Phase A/F1 workflow completed: run_id=${RUN_ID}"
echo "Logs: ${LOG_ROOT}"
