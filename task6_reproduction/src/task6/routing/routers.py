from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from task6.common.config import ARMS
from task6.common.randomness import stream_seed


def stable_topk(scores, k):
    if not 1 <= k <= scores.shape[-1] or not torch.isfinite(scores).all():
        raise ValueError("Invalid top-k scores or budget")
    # Expert columns are already in ascending ID order. Stable descending sort
    # therefore resolves exact ties by the smaller expert ID.
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


def raw_centroids(wi, labels, experts):
    labels = torch.as_tensor(labels, device=wi.device)
    return torch.stack([wi.detach().float()[labels == expert].mean(0) for expert in range(experts)])


class Router(nn.Module):
    """One Phase A router. FFN/backbone parameters remain owned by the model."""

    def __init__(self, arm, centroids, spec, variant, seed, layer):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unsupported Phase A arm: {arm}")
        if seed is None:
            raise ValueError("Every F1 arm requires a training seed")
        self.arm, self.spec = arm, dict(spec)
        self.E, self.D = centroids.shape
        self.aux_weight = float(variant.get("aux_weight", 0.0))
        self.aux = None
        self.register_buffer("pending", torch.zeros(self.E, dtype=torch.long), persistent=False)

        if arm in ("R2", "R2-soft"):
            self.register_buffer("centroids", centroids.detach().float().clone())
        elif arm in ("R4o", "R4d", "R4o-hard"):
            initial = centroids.detach().float().clone()
            if arm in ("R4o", "R4o-hard"):
                generator = torch.Generator(device=initial.device).manual_seed(
                    stream_seed(seed, "summary_init", layer))
                nn.init.orthogonal_(initial, gain=spec["orthogonal_gain"], generator=generator)
            self.summary = nn.Parameter(initial)
        else:
            generator = torch.Generator(device=centroids.device).manual_seed(
                stream_seed(seed, "clean_gate_init", layer))
            bound = 1 / math.sqrt(self.D)
            self.weight = nn.Parameter(torch.empty_like(centroids).uniform_(-bound, bound, generator=generator))
            self.gate_bias = nn.Parameter(
                torch.empty(self.E, device=centroids.device).uniform_(-bound, bound, generator=generator))
            if arm == "G4":
                self.register_buffer("beta", torch.zeros(self.E, device=centroids.device))

    def forward(self, x, k, *, valid):
        self.aux = None
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.arm in ("R2", "R2-soft"):
                eps = self.spec["l2_epsilon"]
                scores = F.normalize(x, dim=-1, eps=eps) @ F.normalize(
                    self.centroids.float(), dim=-1, eps=eps).T
                indices = stable_topk(scores, k)
                weights = k * F.softmax(scores.gather(-1, indices), dim=-1) if self.arm == "R2-soft" else None
                return indices, weights

            if self.arm in ("R4o", "R4d", "R4o-hard"):
                eps = self.spec["rms_epsilon"]
                summary = self.summary.float()
                normalized_x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                normalized_summary = summary * torch.rsqrt(summary.square().mean(-1, keepdim=True) + eps)
                logits = (normalized_x @ normalized_summary.T) / math.sqrt(self.D) / self.spec["temperature"]
                indices = stable_topk(logits, k)
            else:
                logits = F.linear(x, self.weight.float(), self.gate_bias.float())
                probabilities = F.softmax(logits, dim=-1)
                scores = probabilities + self.beta if self.arm == "G4" else logits
                indices = stable_topk(scores, k)
                if self.training and self.arm in ("G2", "G4"):
                    selected = indices[valid]
                    if len(selected) == 0:
                        raise ValueError("No valid tokens in a training layer")
                    counts = torch.bincount(selected.reshape(-1), minlength=self.E)
                    if self.arm == "G2":
                        fraction = counts.float() / (len(selected) * k)
                        self.aux = self.aux_weight * self.E * (fraction * probabilities[valid].mean(0)).sum()
                    else:
                        with torch.no_grad():
                            self.pending.add_(counts)

            weights = k * F.softmax(logits.gather(-1, indices), dim=-1)
            if self.arm == "R4o-hard":
                weights = torch.ones_like(weights) + (weights - weights.detach())
            return indices, weights

    @torch.no_grad()
    def after_step(self):
        if self.arm == "G4" and self.training and self.pending.sum() > 0:
            counts = self.pending.float()
            self.beta.add_(self.spec["bias_update_rate"] * torch.sign(counts.mean() - counts))
        self.pending.zero_()

    @torch.no_grad()
    def clear_pending(self):
        self.pending.zero_()
