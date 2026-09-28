from __future__ import annotations

import numpy as np

from task6.metrics.stability.core import intersections


def overlap_and_coverage(selected, q):
    """Return per-token oracle-set overlap and retained activation mass."""
    selected = np.asarray(selected, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    if selected.ndim != 2 or q.ndim != 2 or len(selected) != len(q):
        raise ValueError("Invalid selection/local-activation shapes")
    if not np.isfinite(q).all() or np.any(q < 0) or q.shape[1] < selected.shape[1]:
        raise ValueError("Invalid local activation sums")
    k = selected.shape[1]
    oracle = np.argsort(-q, axis=1, kind="stable")[:, :k]
    overlap = intersections(selected, oracle).astype(np.float64) / k
    total = q.sum(axis=1, dtype=np.float64)
    keep = total > 0
    selected_mass = np.take_along_axis(q, selected, axis=1).sum(axis=1)
    coverage = selected_mass[keep] / total[keep]
    return overlap, coverage, int((~keep).sum())
