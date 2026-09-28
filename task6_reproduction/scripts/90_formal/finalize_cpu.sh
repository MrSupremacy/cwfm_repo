#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="${1:-configs/local/server.yaml}"
RUN_ID="${2:-main01}"
for stage in aggregate phase-a-check tables figures; do
  "${ROOT}/scripts/run.sh" "${stage}" --local "${LOCAL}" --run-id "${RUN_ID}"
done
