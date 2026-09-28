# Task 8 P0/P1 runbook

Formal locations:

```text
code:    /mnt/luoyulin_code/fanxuankai/task8_reproduction
output:  /mnt/luoyulin_ckpt/fanxuankai/task8_p01
source:  /mnt/luoyulin_code/fanxuankai/task6_phaseD_reproduction
assets:  Task 6 paths or read-only symlinks under inputs/
```

All commands use the existing Task 6 Python environment and add both source trees to `PYTHONPATH` through `scripts/_python.sh`.

```bash
bash scripts/00_preflight/run.sh
bash scripts/10_p0/run_properties.sh
bash scripts/10_p0/freeze_panel.sh
```

One diagnostic smoke (the server setup script resolves the checkpoint id):

```bash
bash scripts/run.sh evaluate --suite configs/suites/p01_init_best.yaml \
  --local configs/local/server.yaml --run-id p01 --population diagnostic_128 \
  --experts 256 --k 51 --role init --mode all

bash scripts/run.sh evaluate --suite configs/suites/p01_init_best.yaml \
  --local configs/local/server.yaml --run-id p01 --population diagnostic_128 \
  --experts 256 --k 51 --role best --seed 0 --mode all
```

Formal full-validation work is sharded by E/k. `M00` is evaluated once per
cell; M01/M10/M11 are evaluated for one init and three best snapshots.

```bash
bash scripts/90_formal/run_p01_8gpu.sh p01_init_best01
bash scripts/30_results/run.sh
```

Outputs are append-safe: an existing result must pass its header and content hashes before it can be reused. The code never overwrites a completed condition.
