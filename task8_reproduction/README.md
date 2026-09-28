# Task 8 P0/P1 and P2-N reproduction

This repository implements the no-training part of Task 8:

- **P0**: asset identity, endpoint compatibility, numerical/property tests, and the frozen `diagnostic_128` panel.
- **P1**: the `selector x aggregation` four-cell intervention (`M00`, `M01`, `M10`, `M11`), full-validation evaluation, fixed-hidden replay kernels, aggregation, tables, and figures.
- **P2-N**: best-only D9 endpoint support/weight profiles (N2-a), fixed-support numeric chains (N1), and fixed-selector temperature interventions (N3). Implementation lives in `src/task8_p2`; see [P2_RUNBOOK.md](P2_RUNBOOK.md) for the eight-GPU launcher and separate result directory.

The repository deliberately reuses Task 6's immutable Dense-MT model, expert splits, data loader, tokenization, and candidate-label scorer. It does **not** copy model/data/checkpoint assets. Task 8 owns the intervention implementation and verifies the Task 6 source tree before evaluation.

The formal protocol uses one seed-deduplicated R4d **init** per E/k plus all
three registered **best** checkpoints. `last/trajectory` are outside the current
P0/P1 execution manifest.

Quick local checks (no model assets required):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m task8_p01 matrix --suite configs/suites/p01_init_best.yaml
```

Server setup and staged commands are in `RUNBOOK.md`.
