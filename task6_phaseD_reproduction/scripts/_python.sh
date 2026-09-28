#!/usr/bin/env bash
set -euo pipefail

# One interpreter/environment boundary for all Linux launchers.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${PYTHON:-}" ]]; then
  INTERPRETER="$PYTHON"
elif [[ -x /mnt/luoyulin_code/fanxuankai/task6_reproduction/.venv/bin/python ]]; then
  # Reuse the already accepted Task6 training environment; do not create another venv.
  INTERPRETER=/mnt/luoyulin_code/fanxuankai/task6_reproduction/.venv/bin/python
else
  INTERPRETER=python
fi
exec "$INTERPRETER" "$@"

