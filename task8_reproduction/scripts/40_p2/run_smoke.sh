#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 <smoke-run-id> [extra smoke options]" >&2; exit 2; }
RUN_ID="$1"
shift
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash scripts/p2.sh smoke --run-id "$RUN_ID" --local configs/local/p2_server.yaml "$@"
