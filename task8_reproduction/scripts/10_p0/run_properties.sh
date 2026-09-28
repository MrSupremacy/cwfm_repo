#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="${1:-p01}"
bash "$ROOT/scripts/run.sh" p0-properties --suite "$ROOT/configs/suites/p01_init_best.yaml" --local "$ROOT/configs/local/server.yaml" --run-id "$RUN_ID"
