#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TASK6_ROOT="${TASK6_ROOT:-/mnt/luoyulin_code/fanxuankai/task6_phaseD_reproduction}"
cd -- "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$TASK6_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${PYTHON:-}" ]]; then
  INTERPRETER="$PYTHON"
elif [[ -x /mnt/luoyulin_code/fanxuankai/task6_reproduction/.venv/bin/python ]]; then
  INTERPRETER=/mnt/luoyulin_code/fanxuankai/task6_reproduction/.venv/bin/python
else
  INTERPRETER=python
fi
exec "$INTERPRETER" "$@"

