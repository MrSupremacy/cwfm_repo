#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 <run-id> [--local-only | --full-only]" >&2; exit 2; }
RUN_ID="$1"
shift
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash scripts/p2.sh launch --run-id "$RUN_ID" --gpus 8 --local configs/local/p2_server.yaml "$@"
