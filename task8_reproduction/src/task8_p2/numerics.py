from __future__ import annotations

import numpy as np


def probabilities(logits, multiplier=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] == 0 or not np.isfinite(logits).all():
        raise ValueError("Expected finite [units, k] selected logits")
    if multiplier == "uniform":
        return np.full_like(logits, 1 / logits.shape[1])
    if not np.isfinite(float(multiplier)) or float(multiplier) <= 0:
        raise ValueError("temperature multiplier must be finite and positive")
    shifted = logits / float(multiplier)
    shifted -= shifted.max(-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(-1, keepdims=True)


def weighting_metrics(logits, p):
    logits, p = np.asarray(logits, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if p.shape != logits.shape or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid probability array")
    if not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=0):
        raise ValueError("Selected probabilities do not sum to one")
    k = p.shape[1]
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(-1)
    variance = np.square(k * p - 1).mean(-1)
    return {
        "selected_logit_std": logits.std(-1, ddof=0),
        "selected_logit_range": np.ptp(logits, axis=-1),
        "weight_variance": variance,
        "n_eff_simpson_over_k": 1 / (k * np.square(p).sum(-1)),
        "entropy_norm": entropy / np.log(k) if k > 1 else np.full(len(p), np.nan),
        "n_eff_entropy_over_k": np.exp(entropy) / k,
        "p_max": p.max(-1), "alpha_max": k * p.max(-1),
    }


def weight_distance(p, reference):
    difference = np.asarray(p) - np.asarray(reference)
    return {"l1": .5 * np.abs(difference).sum(-1),
            "l2": np.sqrt(np.square(difference).sum(-1))}


def boundary(scores, k, atol, rtol, delta):
    scores = np.asarray(scores, dtype=np.float64)
    if not 1 <= k < scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("Boundary diagnostics require 1 <= k < E and finite scores")
    ordered = np.sort(scores, axis=-1)[:, ::-1]
    edge = ordered[:, k-1:k]
    threshold = atol + rtol * np.abs(edge)
    ties = np.abs(scores - edge) <= threshold
    strict = scores > edge + threshold
    gap = ordered[:, k-1] - ordered[:, k]
    return {
        "gap": gap, "gap_norm": gap / (scores.std(-1, ddof=0) + delta),
        "boundary_tie_count": ties.sum(-1),
        "boundary_tie": (ties.sum(-1) > 1).astype(float),
        "strict": strict, "ties": ties,
    }


def support_metrics(ids2, ids4, scores2, scores4, *, atol=1e-6, rtol=1e-6, delta=1e-12):
    ids2, ids4 = np.asarray(ids2), np.asarray(ids4)
    if ids2.shape != ids4.shape:
        raise ValueError("Supports must have the same units and k")
    n, k = ids2.shape
    experts = scores2.shape[1]
    if scores4.shape != (n, experts):
        raise ValueError("Score arrays are not aligned")
    mask2, mask4 = np.zeros((n, experts), bool), np.zeros((n, experts), bool)
    mask2[np.arange(n)[:, None], ids2] = True
    mask4[np.arange(n)[:, None], ids4] = True
    if np.any(mask2.sum(-1) != k) or np.any(mask4.sum(-1) != k):
        raise ValueError("Duplicate expert IDs in top-k support")
    common = (mask2 & mask4).sum(-1)
    b2, b4 = boundary(scores2, k, atol, rtol, delta), boundary(scores4, k, atol, rtol, delta)
    # Maximum common support allowed by each selector's tolerated boundary ties.
    hh = (b2["strict"] & b4["strict"]).sum(-1)
    slack2, slack4 = k-b2["strict"].sum(-1), k-b4["strict"].sum(-1)
    h2t4 = np.minimum((b2["strict"] & b4["ties"]).sum(-1), slack4)
    t2h4 = np.minimum((b2["ties"] & b4["strict"]).sum(-1), slack2)
    tt = np.minimum((b2["ties"] & b4["ties"]).sum(-1),
                    np.minimum(slack2-t2h4, slack4-h2t4))
    return {
        "overlap_at_k": common / k,
        "jaccard": common / (2*k-common), "exact_rate": (common == k).astype(float),
        "replacement_count": k-common, "tie_aware_overlap_at_k": (hh+h2t4+t2h4+tt)/k,
        "gap_R2": b2["gap"], "gap_R4d": b4["gap"],
        "gap_R2_norm": b2["gap_norm"], "gap_R4d_norm": b4["gap_norm"],
        "boundary_tie_count_R2": b2["boundary_tie_count"],
        "boundary_tie_count_R4d": b4["boundary_tie_count"],
        "tie_rate_R2": b2["boundary_tie"], "tie_rate_R4d": b4["boundary_tie"],
        "tie_rate": np.maximum(b2["boundary_tie"], b4["boundary_tie"]),
    }


def chain(q, actual, eta_x, eta_summary, ids, scale):
    """Freeze externally supplied IDs throughout N0–N4; never select here."""
    q = np.take_along_axis(q, ids, -1)
    eta = np.take_along_axis(eta_summary, ids, -1)
    actual = np.take_along_axis(actual, ids, -1)
    values = {
        "N0": np.zeros_like(q), "N1": q, "N2": scale*q,
        "N3": scale*q*eta_x[:, None], "N4": actual,
    }
    if not np.allclose(values["N3"]*eta, actual, atol=3e-5, rtol=3e-5):
        raise ValueError("RMS logits do not match cosine × scale × eta decomposition")
    result = {}
    baseline = probabilities(values["N0"])
    previous = baseline
    for state, logits in values.items():
        p = probabilities(logits)
        metrics = weighting_metrics(logits, p)
        for prefix, reference in (("to_N0", baseline), ("to_previous", previous)):
            for name, value in weight_distance(p, reference).items():
                metrics[f"weight_{name}_{prefix}"] = value
        metrics.update({
            "eta_x": eta_x, "eta_St_mean": eta.mean(-1), "eta_St_variance": eta.var(-1),
            "eta_St_min": eta.min(-1), "eta_St_max": eta.max(-1),
        })
        result[state] = (metrics, p)
        previous = p
    return result


class Moments:
    """Parallel Welford moments; token-weighted mean and sample variance, ddof=1."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite metric cannot silently enter an aggregate")
        count = len(values)
        mean = values.mean(axis=0)
        m2 = np.square(values-mean).sum(axis=0)
        if not self.count:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        delta = mean-self.mean
        total = self.count+count
        self.m2 += m2 + np.square(delta)*self.count*count/total
        self.mean += delta*count/total
        self.count = total

    def values(self):
        if not self.count:
            raise ValueError("No units accumulated")
        variance = self.m2/(self.count-1) if self.count > 1 else np.full_like(self.mean, np.nan)
        return self.mean, variance, np.sqrt(variance)
