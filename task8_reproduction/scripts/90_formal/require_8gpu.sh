#!/usr/bin/env bash
set -euo pipefail
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 1; }
GPU_COUNT="$(nvidia-smi -L | wc -l)"
(( GPU_COUNT >= 8 )) || {
  echo "Eight-GPU launch requires >=8 visible GPUs; found $GPU_COUNT" >&2
  exit 1
}
echo "EIGHT_GPU_PREFLIGHT_OK visible_gpus=$GPU_COUNT"

