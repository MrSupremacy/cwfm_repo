from __future__ import annotations

import numpy as np

from task6_phased.metrics.stability.core import intersections


def selection_statistics(selected, q):
    selected = np.asarray(selected, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    if selected.ndim != 2 or q.ndim != 2 or len(selected) != len(q):
        raise ValueError("Invalid selection/local-activation shapes")
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Local activation sums must be finite and nonnegative")
    k, experts = selected.shape[1], q.shape[1]
    if not 1 <= k < experts:
        raise ValueError("Expected 1 <= k < E")
    oracle = np.argsort(-q, axis=1, kind="stable")[:, :k]
    overlap = intersections(selected, oracle).astype(np.float64) / k
    total = q.sum(axis=1, dtype=np.float64)
    positive = total > 0
    selected_mass = np.take_along_axis(q, selected, axis=1).sum(axis=1, dtype=np.float64)
    coverage = selected_mass[positive] / total[positive]
    return {"overlap": overlap, "coverage": coverage, "all_zero": ~positive}


def overlap_and_coverage(selected, q):
    values = selection_statistics(selected, q)
    return values["overlap"], values["coverage"], int(values["all_zero"].sum())
