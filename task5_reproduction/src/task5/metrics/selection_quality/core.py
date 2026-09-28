from __future__ import annotations

import numpy as np
from task5.common.randomness import stream_seed
from task5.metrics.stability.core import intersections


def overlap_and_coverage(selected, q):
    selected, q = np.asarray(selected).astype(np.int64), np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or len(selected) != len(q) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Invalid local activation sums")
    k = selected.shape[1]
    oracle = np.argsort(-q, axis=1, kind="stable")[:, :k]
    overlap = intersections(selected, oracle).astype(np.float64) / k
    total = q.sum(axis=1, dtype=np.float64)
    keep = total > 0
    coverage = np.take_along_axis(q, selected, axis=1).sum(axis=1)[keep] / total[keep]
    return overlap, coverage, int((~keep).sum())


def expert_pair_matrix(matrix_sum, tokens, labels, experts):
    if tokens <= 0:
        raise ValueError("Empty coactivation population")
    c = np.asarray(matrix_sum, dtype=np.float64) / tokens
    labels = np.asarray(labels)
    if c.shape != (len(labels), len(labels)) or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Invalid coactivation matrix")
    if not np.allclose(c, c.T, rtol=1e-6, atol=1e-5):
        raise ValueError("Asymmetric coactivation matrix")
    sizes = np.bincount(labels, minlength=experts)
    if np.any(sizes != sizes[0]) or len(sizes) != experts:
        raise ValueError("Expected equal-size expert labels")
    order = np.argsort(labels, kind="stable")
    return c[np.ix_(order, order)].reshape(experts, sizes[0], experts, sizes[0]).mean(axis=(1, 3))


def pair_scores(selected, pair_matrix):
    selected = np.asarray(selected, dtype=np.int64)
    if selected.shape[1] < 2:
        raise ValueError("Coactivation consistency requires at least two experts")
    i, j = np.triu_indices(selected.shape[1], 1)
    return pair_matrix[selected[:, i], selected[:, j]].mean(axis=1, dtype=np.float64)


def random_reference(pair_matrix, tokens, k, repeats, seed, identity, chunk_rows=8192):
    """One independent uniform-rank sample per token; RNG consumption is chunk invariant."""
    if tokens <= 0 or repeats <= 0 or chunk_rows <= 0 or not 2 <= k <= len(pair_matrix):
        raise ValueError("Empty random reference")
    values = []
    E = len(pair_matrix)
    for repeat in range(repeats):
        rng = np.random.Generator(np.random.PCG64(stream_seed(seed, identity, repeat)))
        def scores():
            for start in range(0, tokens, chunk_rows):
                ranks = rng.random((min(chunk_rows, tokens - start), E))
                selected = np.argsort(ranks, axis=1, kind="stable")[:, :k]
                yield from pair_scores(selected, pair_matrix)
        import math
        values.append(math.fsum(scores()) / tokens)
    return np.asarray(values, dtype=np.float64)


def consistency_summary(selected_mean, repeats, epsilon=1e-12):
    repeats = np.asarray(repeats, dtype=np.float64)
    random_mean = float(repeats.mean())
    low, high = np.quantile(repeats, [0.025, 0.975], method="linear")
    return {"selected_mean": float(selected_mean), "random_mean": random_mean,
            "random_low": float(low), "random_high": float(high),
            "excess": float(selected_mean - random_mean),
            "ratio": float(selected_mean / random_mean) if random_mean > epsilon else None,
            "random_replicates": repeats.tolist()}
