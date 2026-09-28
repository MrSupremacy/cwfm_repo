#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <run-id>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SUITE=configs/suites/p01_init_best.yaml
LOCAL=configs/local/server.yaml
COMMON=(--suite "$SUITE" --local "$LOCAL" --run-id "$RUN_ID")

bash scripts/05_assets/link_server_assets.sh
bash scripts/10_p0/run_properties.sh "$RUN_ID"
bash scripts/00_preflight/run.sh "$RUN_ID"
bash scripts/10_p0/freeze_panel.sh

# Representative real-model endpoint checks for both retained roles.
bash scripts/run.sh evaluate "${COMMON[@]}" --population diagnostic_128 \
  --experts 256 --k 51 --role init --mode all
bash scripts/run.sh compare-endpoints "${COMMON[@]}" \
  --experts 256 --k 51 --role init
bash scripts/run.sh evaluate "${COMMON[@]}" --population diagnostic_128 \
  --experts 256 --k 51 --role best --seed 0 --mode all
bash scripts/run.sh compare-endpoints "${COMMON[@]}" \
  --experts 256 --k 51 --role best --seed 0
echo "TASK8_P0_COMPLETE run_id=$RUN_ID"

