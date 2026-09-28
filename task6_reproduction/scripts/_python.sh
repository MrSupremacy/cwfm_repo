#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
if [[ -n "${PYTHON:-}" ]]; then
  EXECUTABLE="${PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  EXECUTABLE="${REPO_ROOT}/.venv/bin/python"
else
  EXECUTABLE="python"
fi
exec "${EXECUTABLE}" "$@"
