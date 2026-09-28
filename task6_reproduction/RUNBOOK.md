# Task 6 Phase A/F1 runbook

All commands run from the repository root. `RUN_ID` names one immutable run family. Formal commands use `configs/suites/phase_a_f1.yaml` implicitly and load machine paths from `configs/local/server.yaml`.

The launchers resolve Python in this order: an explicit `$PYTHON`, then the repository `.venv`, then `python` on `PATH`. If the cluster environment already contains the pinned packages, point to it and do not create another environment:

```bash
export PYTHON=/absolute/path/to/the/existing/environment/bin/python
```

Only when no prepared environment exists, create an isolated fallback. `--system-site-packages` reuses the server's CUDA PyTorch build while keeping project dependencies out of system Python:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -c configs/local/server-tested.txt -e '.[model]'
```

## 1. Gates before training

```bash
scripts/20_validate/test_local.sh
scripts/00_preflight/run.sh configs/local/server.yaml main01
scripts/10_prepare/run.sh configs/local/server.yaml main01
scripts/20_validate/run.sh configs/local/server.yaml main01
```

`preflight` must report 198 training runs, 2178 full checkpoints and 2180 A captures: 192 routed runs, six task-by-seed `Dense-fullFT` runs and two fixed `Dense-init` references. `validate` loads real assets, checks exact all-expert equivalence for every routed arm, and verifies fullFT backbone/router gradient paths. The local smoke test remains useful after code or protocol changes; repeating the expensive real-input preflight is only required before launching a newly modified formal run.

## 2. One-condition rehearsal

Use a fresh non-formal run ID and stop at the first epoch boundary:

```bash
scripts/30_train/run.sh configs/local/server.yaml rehearsal01 \
  --task mnli --arm R4o-hard --k 26 --seed 0 --stop-after-epoch 1
```

This real-data rehearsal writes a full step-0 and step-1 checkpoint. It is separate from the formal result family.

## 3. Formal training

### One-command complete workflow

On the eight-GPU production server, a fresh run can execute preflight, prepare, Phase 0, all training, A, best selection, B/C+D, metrics, aggregation, checks, tables and figures with:

```bash
export PYTHON=/absolute/path/to/the/existing/environment/bin/python
bash scripts/90_formal/run_all_8gpu.sh configs/local/server.yaml main01
```

To append the read-only Task 5 F0 comparison after F1 completes, pass its normalized result as the third argument:

```bash
bash scripts/90_formal/run_all_8gpu.sh configs/local/server.yaml main01 \
  /absolute/path/to/task5/results/data/normalized/RUN_ID/metrics.json
```

The script requires exactly eight visible GPUs. Training, A capture, diagnostic capture and metrics use eight condition shards; prepare/Phase 0 and final aggregation run once. Every stage waits for all shards and stops on failure. Logs are written under `tmp/formal_launch/<run_id>/`.

### Training-only launcher

```bash
scripts/90_formal/run_single_node_8gpu.sh configs/local/server.yaml main01
```

The formal launcher requires exactly eight visible GPUs and assigns one independent condition shard to each. Every run still has `world_size=1`; this is not DDP. A run directory is created exclusively, so rerunning without `--resume` fails rather than overwriting it. Resume one condition with the normal train entry and an explicit checkpoint name.

## 4. Capture and analysis

Capture can be partitioned across GPUs with the same `--shard-index/--shard-count` arguments. Run stages in this order:

```bash
scripts/40_capture/run.sh configs/local/server.yaml main01 --part A
scripts/40_capture/run.sh configs/local/server.yaml main01 --part select-best
scripts/40_capture/run.sh configs/local/server.yaml main01 --part diagnostics
scripts/50_metrics/run.sh configs/local/server.yaml main01
scripts/90_formal/finalize_cpu.sh configs/local/server.yaml main01
```

`A` covers all 2178 trained checkpoints and two `Dense-init` references. `Dense-fullFT` receives performance evaluation and best-checkpoint selection only. `B` covers each routed run's deduplicated best/final states, and `C+D` covers all 2112 routed checkpoints. Metrics for routed models are performance, CV, churn/exact-set-change, oracle overlap and activation coverage; dense models only use performance metrics.

## 5. Add Dense-fullFT to an earlier routed-only run

Do not load checkpoints from an already-running 192-condition experiment with this revised 198-condition protocol: the protocol hashes intentionally differ. Let the original checkout finish its sparse capture, metrics and aggregation first. Then run only the six matched dense conditions and combine the lightweight results without changing the old output:

```bash
bash scripts/90_formal/run_dense_fullft_extension_8gpu.sh \
  configs/local/server.yaml denseft01 \
  /absolute/path/to/old/output/results old_sparse_run_id
```

This launcher first runs the revised local tests and a config-only matrix check. It then uses GPUs 0–5 for the six task-by-seed runs, captures a fresh protocol-matched Dense-init denominator, and computes only dense performance. It copies the completed routed-only tables and figures into a new result directory, then replaces only the main best-validation table and the accuracy/relative-performance figures. CV, churn, overlap and coverage figures plus diagnostic/appendix tables remain byte-for-byte copies of the routed-only outputs.

```text
<output_root>/results/dense_fullft_extension/denseft01/
```

The old normalized, aggregated and paired-difference files are read-only inputs. Their paths, SHA256 hashes and original metadata are recorded in the combined result. The original result directory is never overwritten.

## 6. Output safety

The formal output root is outside this repository. Checkpoints contain `model_state.pt`, `training_state.pt`, `meta.json` and a hash manifest `complete.json`. No stage automatically selects the latest run, replaces a complete checkpoint, removes old results, or starts distributed data parallel training.
