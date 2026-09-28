#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 LOCAL_CONFIG RUN_ID TASK5_F0_NORMALIZED_JSON" >&2
  exit 2
fi
exec "${ROOT}/scripts/run.sh" compare-f0 --local "$1" --run-id "$2" --f0-results "$3"
