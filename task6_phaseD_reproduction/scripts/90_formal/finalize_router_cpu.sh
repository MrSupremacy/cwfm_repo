#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <routed-run-id>" >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
COMMON=(--suite configs/suites/phase_d_f0.yaml --local configs/local/server.yaml --run-id "$1")
bash scripts/run.sh aggregate "${COMMON[@]}"
bash scripts/run.sh tables "${COMMON[@]}"
bash scripts/run.sh figures "${COMMON[@]}"

