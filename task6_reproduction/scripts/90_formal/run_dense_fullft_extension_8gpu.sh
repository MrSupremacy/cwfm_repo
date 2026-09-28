#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${1:-configs/local/server.yaml}"
DENSE_RUN_ID="${2:-}"
SPARSE_RESULT_ROOT="${3:-}"
SPARSE_RUN_ID="${4:-}"

if [[ -z "${DENSE_RUN_ID}" || -z "${SPARSE_RESULT_ROOT}" || -z "${SPARSE_RUN_ID}" ]]; then
  echo "usage: $0 LOCAL_CONFIG DENSE_RUN_ID SPARSE_RESULTS_ROOT SPARSE_RUN_ID" >&2
  exit 2
fi

cd "${ROOT}"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
  echo "Dense-fullFT extension requires exactly 8 visible GPUs; found ${GPU_COUNT}" >&2
  exit 1
fi

LOG_ROOT="${ROOT}/tmp/dense_fullft_extension/${DENSE_RUN_ID}"
mkdir -p "${LOG_ROOT}"

"${ROOT}/scripts/20_validate/test_local.sh" >"${LOG_ROOT}/tests.log" 2>&1
"${ROOT}/scripts/run.sh" preflight --config-only --local "${LOCAL}" \
  --run-id "${DENSE_RUN_ID}" >"${LOG_ROOT}/matrix.log" 2>&1
"${ROOT}/scripts/run.sh" prepare --local "${LOCAL}" --run-id "${DENSE_RUN_ID}" \
  >"${LOG_ROOT}/prepare.log" 2>&1

pids=()
index=0
for task in sst2 mnli; do
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES="${index}" "${ROOT}/scripts/run.sh" train \
      --local "${LOCAL}" --run-id "${DENSE_RUN_ID}" \
      --task "${task}" --arm dense-ft --seed "${seed}" \
      >"${LOG_ROOT}/train_${task}_seed_${seed}.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

pids=()
for task in sst2 mnli; do
  CUDA_VISIBLE_DEVICES=0 "${ROOT}/scripts/run.sh" capture --part A \
    --local "${LOCAL}" --run-id "${DENSE_RUN_ID}" --task "${task}" --arm dense \
    >"${LOG_ROOT}/capture_dense_init_${task}.log" 2>&1
done
index=0
for task in sst2 mnli; do
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES="${index}" "${ROOT}/scripts/run.sh" capture --part A \
      --local "${LOCAL}" --run-id "${DENSE_RUN_ID}" \
      --task "${task}" --arm dense-ft --seed "${seed}" \
      >"${LOG_ROOT}/capture_${task}_seed_${seed}.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${ROOT}/scripts/run.sh" capture --part select-best --local "${LOCAL}" \
  --run-id "${DENSE_RUN_ID}" --arm dense-ft >"${LOG_ROOT}/select_best.log" 2>&1
"${ROOT}/scripts/run.sh" metrics --metric performance --local "${LOCAL}" \
  --run-id "${DENSE_RUN_ID}" --arm dense-ft >"${LOG_ROOT}/metrics.log" 2>&1
"${ROOT}/scripts/run.sh" dense-extension-report --local "${LOCAL}" \
  --run-id "${DENSE_RUN_ID}" --sparse-result-root "${SPARSE_RESULT_ROOT}" \
  --sparse-run-id "${SPARSE_RUN_ID}" >"${LOG_ROOT}/report.log" 2>&1

echo "Dense-fullFT extension completed: ${DENSE_RUN_ID}"
echo "Logs: ${LOG_ROOT}"
