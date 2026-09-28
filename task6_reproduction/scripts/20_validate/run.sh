#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT}/scripts/run.sh" validate --local "${1:-configs/local/server.yaml}" --run-id "${2:-main01}" "${@:3}"
