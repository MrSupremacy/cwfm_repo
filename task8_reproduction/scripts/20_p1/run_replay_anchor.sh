#!/usr/bin/env bash
set -euo pipefail
EXPERTS="${1:?experts required}"
K="${2:?k required}"
ROLE="${3:?role init-or-best required}"
SEED="${4:-}"
RUN_ID="${5:-p01}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ARGS=(replay --suite "$ROOT/configs/suites/p01_init_best.yaml" --local "$ROOT/configs/local/server.yaml" \
  --run-id "$RUN_ID" --experts "$EXPERTS" --k "$K" --role "$ROLE")
if [[ "$ROLE" == best ]]; then
  [[ -n "$SEED" ]] || { echo "best requires seed" >&2; exit 2; }
  ARGS+=(--seed "$SEED")
elif [[ "$ROLE" != init || -n "$SEED" ]]; then
  echo "init must omit seed" >&2
  exit 2
fi
bash "$ROOT/scripts/run.sh" "${ARGS[@]}"
