#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <dense-run-id>" >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
COMMON=(--suite configs/suites/dense_mt.yaml --local configs/local/server.yaml --run-id "$1")
bash scripts/run.sh select-dense-best "${COMMON[@]}"
bash scripts/run.sh dense-results "${COMMON[@]}"

