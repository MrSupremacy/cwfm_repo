#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
TASK6_ROOT=/mnt/luoyulin_code/fanxuankai/task6_phaseD_reproduction
link_once() {
  local source="$1" target="$2"
  if [[ -L "$target" ]]; then
    [[ "$(readlink -f -- "$target")" == "$(readlink -f -- "$source")" ]] || { echo "mismatched link: $target" >&2; exit 1; }
  elif [[ -e "$target" ]]; then
    echo "refusing to replace non-link: $target" >&2
    exit 1
  else
    ln -s -- "$source" "$target"
  fi
}
link_once "$TASK6_ROOT/inputs/dense_mt/best" "$ROOT/inputs/dense_best"
link_once "$TASK6_ROOT/inputs/data" "$ROOT/inputs/data"
link_once "$TASK6_ROOT/inputs/expert_splits" "$ROOT/inputs/expert_splits"
bash "$ROOT/scripts/run.sh" build-catalog --suite "$ROOT/configs/suites/p01_init_best.yaml" --local "$ROOT/configs/local/server.yaml" --create-links
