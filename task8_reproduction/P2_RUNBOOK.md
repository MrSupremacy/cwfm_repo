# Task 8 P2-N execution

This extends the existing repository with **N2-a, N1 and N3 only**, using D9 best seeds 0/1/2. No training is performed. The controlling design is `task8_analysis/P2/02_task8_p2_n_forward_experiment_logic.md`.

## Locations and assets

- Code: `/mnt/luoyulin_code/fanxuankai/task8_reproduction`.
- P2 output: `/mnt/luoyulin_ckpt/fanxuankai/task8_p2_n`.
- P1 read-only source: `/mnt/luoyulin_ckpt/fanxuankai/task8_p01`.
- Models, summaries, datasets and expert splits are referenced through the existing Task6 configuration and P1 checkpoint catalog. They are not copied into a new asset store.
- `task8_p01` source remains unchanged. P2 is implemented in `src/task8_p2`.

## Local checks without assets

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src/task8_p2
PYTHONPATH=src python -m task8_p2 matrix
```

PyYAML and NumPy are needed for the asset-free checks. PyTorch-dependent tests can skip locally; all tests must pass on the development machine with the existing Task6 environment.

## Formal eight-GPU run

```bash
cd /mnt/luoyulin_code/fanxuankai/task8_reproduction
bash scripts/90_formal/run_p2_n_8gpu.sh p2_n_forward01
```

The launcher checks eight visible CUDA devices, preserves the scheduler's `CUDA_VISIBLE_DEVICES` mapping, then runs:

1. Preflight: data/checkpoint hashes and all 54 P1 M10/M11 endpoints.
2. Prepare: one Dense trace and nine seed-independent natural R2 traces.
3. Eight workers: 27 natural R4d traces and local N2-a/N1/N3 diagnostics.
4. Eight workers: 81 new full evaluations (27 snapshots × m=2/4/8).
5. Merge tables, reuse m=1/uniform endpoints, render separate P2 PNG/PDF figures.

Progress bars are line-based and flushed, so they remain visible in scheduler logs. They report stage, worker, task/batch or snapshot count, elapsed time and ETA. Model loading, hashing and rendering also emit heartbeats every 20 seconds. Worker logs are retained under `task8_p2_n/logs/<run-id>/`.

Run local diagnostics first if desired:

```bash
bash scripts/90_formal/run_p2_n_8gpu.sh p2_n_local01 --local-only
bash scripts/90_formal/run_p2_n_8gpu.sh p2_n_forward01 --full-only
```

Use different report IDs for the local-only report and the final report. Per-job artifacts are shared and verified on resume; a final report ID is immutable.

## Single-GPU smoke and individual jobs

```bash
# E64/k6, E128/k26, E256/k51, seed0; two samples per task.
bash scripts/40_p2/run_smoke.sh p2_n_smoke01 --samples-per-task 2

# One smoke anchor; never enters the formal result namespace.
bash scripts/40_p2/run_smoke.sh p2_n_smoke_e256 --experts 256 --k 51 --seed 0 --samples-per-task 2

# One formal diagnostic job: 128 frozen samples per task, all 12 layers.
bash scripts/p2.sh local --experts 256 --k 51 --seed 0 --local configs/local/p2_server.yaml

# One formal full-eval job: m=2/4/8. Requires that snapshot's local checks.
bash scripts/p2.sh full --experts 256 --k 51 --seed 0 --local configs/local/p2_server.yaml

# Rebuild the final report after all 27 snapshots are complete.
bash scripts/p2.sh results --run-id p2_n_forward01 --local configs/local/p2_server.yaml
```

Smoke additionally recomputes tiny m=1 and uniform predictions and checks them against the corresponding subset of P1 full-validation predictions (zero prediction mismatch; candidate-score tolerance 0.01). Formal m=1/uniform are reused, not reevaluated.

## Output and interpretation

```text
task8_p2_n/
  artifacts/preflight/<protocol>/
  caches/<protocol>/<diagnostic-or-smoke-role>/traces/
    dense/shared/
    natural_R2/E_*/k_*/static/
    natural_R4d/E_*/k_*/seed_*/best/
  runs/local/<protocol>/E_*/k_*/seed_*/best/
    raw/                         # pickle-free NPZ unit records
    p2_*.csv                     # scoped metrics / rank moments
    checks.json
  runs/evaluation/<protocol>/E_*/k_*/seed_*/best/m_{2,4,8}/
    {sst2,mnli,qnli,qqp}.json.gz # predictions and candidate scores
  runs/smoke_local/...
  runs/smoke_evaluation/...
  results/<run-id>/
    tables/
    figures/                     # 15 local-only or 22 full PNG/PDF figures
    manifest.json
    complete.json
  logs/<run-id>/
```

Main rank curves pool non-padding token-layer units. Their variance uses ddof=1; shaded bands are unit SD, not seed SD or standard error. Per-task/layer/stack/token-group rows remain available. `p` sums to one; `alpha=k*p` sums to k. Simpson `n_eff_simpson_over_k` and entropy `n_eff_entropy_over_k` are separately named.

`p2_local_units_index.csv` maps raw unit arrays to exact trace files and chunk hashes. Within each raw NPZ, `sample_id`, `token_position`, and `token_group` align all arrays; `*_metric_names` define the columns in `*_values`; `*_p` are unit-by-k probabilities. A hash identifies a complete aligned hidden chunk, not an independently hashed token.

N1 and local N3 never retop-k an externally supplied support. Full-model N3 uses the original selector on each condition's current hidden; later hidden and support may differ across m because earlier outputs differ. The performance-vs-variance scatter uses local statistics on cached original M11 hidden and labels that distinction explicitly.

Artifacts publish atomically after streams close and hashes finish. Resume verifies identity and contents, and never silently overwrites an existing incompatible artifact. Failed work is retained in a hidden `.partial-*` sibling for inspection, not merged. Advisory locks release on process exit. Code/protocol changes use another namespace. No P0/P1 result is modified.

Temperature is not selected automatically. The design's proposed P3 selection rule still needs a quantitative definition of “close”/tie handling before a temperature can be locked for training.
