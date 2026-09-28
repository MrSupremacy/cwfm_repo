#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <run-id>" >&2; exit 2; }
RUN_ID="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
bash scripts/90_formal/require_8gpu.sh
SPECS=()
for cell in 64:6 256:26 256:51 128:26; do
  IFS=: read -r experts k <<<"$cell"
  SPECS+=("$experts:$k:init:")
  for seed in 0 1 2; do SPECS+=("$experts:$k:best:$seed"); done
done
pids=()
for shard in {0..7}; do
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    for index in "${!SPECS[@]}"; do
      (( index % 8 == shard )) || continue
      IFS=: read -r experts k role seed <<<"${SPECS[$index]}"
      if [[ "$role" == init ]]; then
        bash scripts/20_p1/run_replay_anchor.sh "$experts" "$k" init "" "$RUN_ID"
      else
        bash scripts/20_p1/run_replay_anchor.sh "$experts" "$k" best "$seed" "$RUN_ID"
      fi
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]]
echo "TASK8_P1_REPLAY_COMPLETE run_id=$RUN_ID"
