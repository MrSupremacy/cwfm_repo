from __future__ import annotations

import itertools
import math
import numpy as np


def utilization(selected, experts):
    values = np.asarray(selected, dtype=np.int64)
    counts = np.bincount(values.reshape(-1), minlength=experts).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("Empty utilization population")
    return counts / counts.sum(), counts.astype(np.int64)


def js_distance(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    middle = (left + right) / 2
    def kl(value):
        positive = value > 0
        return np.sum(value[positive] * np.log2(value[positive] / middle[positive]))
    return float(math.sqrt(max(0.0, 0.5 * kl(left) + 0.5 * kl(right))))


def domain_metrics(distributions, task_order):
    matrix = np.stack([distributions[task] for task in task_order])
    marginal = matrix.mean(0)
    terms = np.zeros_like(matrix)
    positive = matrix > 0
    terms[positive] = matrix[positive] * np.log2(
        matrix[positive] / np.broadcast_to(marginal, matrix.shape)[positive]
    )
    mi = float(terms.sum() / len(task_order))
    pairs = {
        f"{left}__{right}": js_distance(distributions[left], distributions[right])
        for left, right in itertools.combinations(task_order, 2)
    }
    return {"pairwise_js_distance": pairs, "mean_js_distance": float(np.mean(list(pairs.values()))),
            "max_js_distance": float(np.max(list(pairs.values()))), "mi_bits": mi, "nmi_domain": mi / 2.0}
