#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
bash "$ROOT/scripts/run.sh" freeze-panel --suite "$ROOT/configs/suites/p01_init_best.yaml" --local "$ROOT/configs/local/server.yaml"
