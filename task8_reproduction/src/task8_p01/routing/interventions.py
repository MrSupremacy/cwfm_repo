from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


MODES = ("M00", "M01", "M10", "M11")


def _matrix(name, value):
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [experts, d_model]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    return value.float()


def stable_topk(scores, k):
    if scores.ndim < 2 or not 1 <= int(k) <= scores.shape[-1]:
        raise ValueError("Invalid top-k score shape or budget")
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite routing scores")
    # Stable descending sort gives ascending expert id for exact ties.
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., : int(k)]


def l2_logits(hidden, centroids, epsilon=1.0e-12):
    centroids = _matrix("centroids", centroids)
    x = hidden.float()
    if x.shape[-1] != centroids.shape[-1] or not torch.isfinite(x).all():
        raise ValueError("Hidden/centroid shape mismatch or non-finite hidden")
    return functional.normalize(x, dim=-1, eps=float(epsilon)) @ functional.normalize(
        centroids, dim=-1, eps=float(epsilon)
    ).T


def rms_logits(hidden, summaries, epsilon=1.0e-6, temperature=1.0):
    summaries = _matrix("summaries", summaries)
    x = hidden.float()
    if x.shape[-1] != summaries.shape[-1] or not torch.isfinite(x).all():
        raise ValueError("Hidden/summary shape mismatch or non-finite hidden")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    x_rms = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + float(epsilon))
    summary_rms = summaries * torch.rsqrt(
        summaries.square().mean(-1, keepdim=True) + float(epsilon)
    )
    return (x_rms @ summary_rms.T) / (math.sqrt(summaries.shape[-1]) * float(temperature))


def selector(hidden, selector_state, k):
    """Return only support indices; selection and weighting are intentionally separate."""
    kind = selector_state["kind"]
    if kind == "r2_l2":
        scores = l2_logits(hidden, selector_state["summaries"], selector_state["epsilon"])
    elif kind == "r4d_rms":
        scores = rms_logits(
            hidden,
            selector_state["summaries"],
            selector_state["epsilon"],
            selector_state.get("temperature", 1.0),
        )
    else:
        raise ValueError(f"Unknown selector: {kind}")
    return stable_topk(scores, k)


def weighter(hidden, weight_state, supplied_indices):
    """Weight an externally supplied support. This function never performs top-k."""
    if supplied_indices.ndim != hidden.ndim or supplied_indices.shape[:-1] != hidden.shape[:-1]:
        raise ValueError("supplied_indices is not aligned with hidden")
    experts = int(weight_state["summaries"].shape[0])
    if supplied_indices.dtype != torch.long:
        raise ValueError("supplied_indices must be torch.long")
    if supplied_indices.numel() and (
        int(supplied_indices.min()) < 0 or int(supplied_indices.max()) >= experts
    ):
        raise ValueError("supplied_indices contains an invalid expert id")
    kind = weight_state["kind"]
    if kind == "uniform":
        return torch.ones_like(supplied_indices, dtype=hidden.dtype)
    if kind != "r4d_soft":
        raise ValueError(f"Unknown weighter: {kind}")
    logits = rms_logits(
        hidden,
        weight_state["summaries"],
        weight_state["epsilon"],
        weight_state.get("temperature", 1.0),
    )
    selected_logits = logits.gather(-1, supplied_indices)
    k = supplied_indices.shape[-1]
    return int(k) * functional.softmax(selected_logits, dim=-1)


def aggregate(expert_outputs, supplied_indices, weights):
    """Combine `[... E, D]` expert outputs over the supplied support."""
    if expert_outputs.ndim < 3:
        raise ValueError("expert_outputs must have shape [..., experts, d_model]")
    if supplied_indices.shape != weights.shape:
        raise ValueError("indices and weights must have identical shapes")
    if expert_outputs.shape[:-2] != supplied_indices.shape[:-1]:
        raise ValueError("expert outputs and routing rows are not aligned")
    gather_index = supplied_indices.unsqueeze(-1).expand(*supplied_indices.shape, expert_outputs.shape[-1])
    chosen = expert_outputs.gather(-2, gather_index)
    return (chosen * weights.unsqueeze(-1).to(chosen.dtype)).sum(-2)


def mode_states(mode, c0, learned, *, l2_epsilon=1.0e-12, rms_epsilon=1.0e-6, temperature=1.0):
    if mode not in MODES:
        raise ValueError(f"Unknown four-cell mode: {mode}")
    if c0.shape != learned.shape:
        raise ValueError("C0 and learned summaries must have identical shapes")
    use_r4_selector = mode in ("M10", "M11")
    use_soft = mode in ("M01", "M11")
    selected = {
        "kind": "r4d_rms" if use_r4_selector else "r2_l2",
        "summaries": learned if use_r4_selector else c0,
        "epsilon": rms_epsilon if use_r4_selector else l2_epsilon,
        "temperature": temperature,
    }
    weighted = {
        "kind": "r4d_soft" if use_soft else "uniform",
        # M01 must use S_t logits even though its support came from C0/R2.
        "summaries": learned,
        "epsilon": rms_epsilon,
        "temperature": temperature,
    }
    return selected, weighted


def route(
    mode,
    hidden,
    c0,
    learned,
    k,
    *,
    l2_epsilon=1.0e-12,
    rms_epsilon=1.0e-6,
    temperature=1.0,
):
    selected_state, weight_state = mode_states(
        mode,
        c0,
        learned,
        l2_epsilon=l2_epsilon,
        rms_epsilon=rms_epsilon,
        temperature=temperature,
    )
    indices = selector(hidden, selected_state, k)
    weights = weighter(hidden, weight_state, indices)
    return indices, weights

