from __future__ import annotations

import numpy as np

from task6_phaseb.metrics.stability.core import intersections


def selection_statistics(selected, q):
    selected = np.asarray(selected, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    if selected.ndim != 2 or q.ndim != 2 or len(selected) != len(q):
        raise ValueError("Invalid selection/local-activation shapes")
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Local activation sums must be finite and nonnegative")
    k, experts = selected.shape[1], q.shape[1]
    if not 1 <= k < experts:
        raise ValueError("Oracle boundary metrics require 1 <= k < E")
    order = np.argsort(-q, axis=1, kind="stable")
    oracle = order[:, :k]
    overlap = intersections(selected, oracle).astype(np.float64) / k
    random_expectation = k / experts
    adjusted = (overlap - random_expectation) / (1 - random_expectation)
    total = q.sum(axis=1, dtype=np.float64)
    positive = total > 0
    selected_mass = np.take_along_axis(q, selected, axis=1).sum(axis=1, dtype=np.float64)
    coverage = selected_mass[positive] / total[positive]
    sorted_q = np.take_along_axis(q, order[:, :k + 1], axis=1)
    boundary_gap = sorted_q[:, k - 1] - sorted_q[:, k]
    normalized_gap = boundary_gap[positive] / total[positive]
    return {
        "overlap": overlap,
        "adjusted_overlap": adjusted,
        "coverage": coverage,
        "boundary_gap": boundary_gap,
        "normalized_boundary_gap": normalized_gap,
        "boundary_tie": boundary_gap == 0,
        "all_zero": ~positive,
    }


def overlap_and_coverage(selected, q):
    values = selection_statistics(selected, q)
    return values["overlap"], values["coverage"], int(values["all_zero"].sum())
