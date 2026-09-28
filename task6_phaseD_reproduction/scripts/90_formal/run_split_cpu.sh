#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
COMMON=(--suite configs/suites/phase_d_f0.yaml --local configs/local/server.yaml --run-id split_seed1)
for experts in 128 256; do
  bash scripts/run.sh generate-split "${COMMON[@]}" --experts "$experts"
  bash scripts/run.sh validate-split "${COMMON[@]}" --experts "$experts"
done
echo "PHASE_D_SPLITS_READY experts=128,256 random_state=1"
