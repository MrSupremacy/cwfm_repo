#!/usr/bin/env bash
set -euo pipefail
SHARD_INDEX="${1:-0}"
SHARD_COUNT="${2:-1}"
RUN_ID="${3:-p01}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CELLS=(64:6 64:10 64:13 128:13 128:19 128:26 256:26 256:38 256:51)
for index in "${!CELLS[@]}"; do
  (( index % SHARD_COUNT == SHARD_INDEX )) || continue
  IFS=: read -r experts k <<<"${CELLS[$index]}"
  # Keep init and all best seeds of one cell on the same shard. Init creates
  # M00; best snapshots validate/reuse it, avoiding cross-process races.
  bash "$ROOT/scripts/run.sh" evaluate --suite "$ROOT/configs/suites/p01_init_best.yaml" \
    --local "$ROOT/configs/local/server.yaml" --run-id "$RUN_ID" \
    --population full_validation --experts "$experts" --k "$k" --role init --mode all
  for seed in 0 1 2; do
    bash "$ROOT/scripts/run.sh" evaluate --suite "$ROOT/configs/suites/p01_init_best.yaml" \
      --local "$ROOT/configs/local/server.yaml" --run-id "$RUN_ID" \
      --population full_validation --experts "$experts" --k "$k" \
      --role best --seed "$seed" --mode all
  done
done
