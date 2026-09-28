from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from task6_phased.common.randomness import stream_seed


def stable_topk(scores, k):
    if not 1 <= k <= scores.shape[-1] or not torch.isfinite(scores).all():
        raise ValueError("Invalid top-k scores/budget")
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


def raw_centroids(wi, labels, experts):
    values = wi.detach().float()
    return torch.stack([values[labels == expert].mean(0) for expert in range(experts)])


class Router(nn.Module):
    """One Phase D/F0 router; frozen FFN weights remain owned by MaskedFFN."""

    def __init__(self, arm, centroids, spec, variant, seed, layer):
        super().__init__()
        self.arm = arm
        self.spec = dict(spec)
        self.E, self.D = centroids.shape
        self.aux_weight = float(variant.get("aux_weight", 0.0))
        self.aux = None
        self.last_stats = None
        self.last_update = None
        self.register_buffer("pending", torch.zeros(self.E, dtype=torch.long), persistent=False)

        if arm == "R2":
            self.register_buffer("centroids", centroids.clone())
        elif arm in ("R4o", "R4d"):
            initial = centroids.clone()
            if arm == "R4o":
                generator = torch.Generator(device=initial.device).manual_seed(
                    stream_seed(seed, "summary_init", layer)
                )
                nn.init.orthogonal_(initial, gain=spec["orthogonal_gain"], generator=generator)
            self.summary = nn.Parameter(initial)
        elif arm in ("G1", "G2-0.001", "G4"):
            generator = torch.Generator(device=centroids.device).manual_seed(
                stream_seed(seed, "clean_gate_init", layer)
            )
            bound = 1 / math.sqrt(self.D)
            self.weight = nn.Parameter(
                torch.empty_like(centroids).uniform_(-bound, bound, generator=generator)
            )
            self.gate_bias = nn.Parameter(
                torch.empty(self.E, device=centroids.device).uniform_(-bound, bound, generator=generator)
            )
            if arm == "G4":
                self.register_buffer("beta", torch.zeros(self.E, device=centroids.device))
        else:
            raise ValueError(f"Unsupported Phase D arm: {arm}")

    def forward(self, x, k, *, valid):
        self.aux = None
        self.last_stats = None
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.arm == "R2":
                epsilon = self.spec["l2_epsilon"]
                scores = F.normalize(x, dim=-1, eps=epsilon) @ F.normalize(
                    self.centroids.float(), dim=-1, eps=epsilon
                ).T
                return stable_topk(scores, k), None

            if self.arm in ("R4o", "R4d"):
                epsilon = self.spec["rms_epsilon"]
                summary = self.summary.float()
                normalized_x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
                normalized_summary = summary * torch.rsqrt(
                    summary.square().mean(-1, keepdim=True) + epsilon
                )
                logits = (normalized_x @ normalized_summary.T) / math.sqrt(self.D)
                logits = logits / self.spec["temperature"]
                indices = stable_topk(logits, k)
            else:
                logits = F.linear(x, self.weight.float(), self.gate_bias.float())
                probabilities = F.softmax(logits, dim=-1)
                selection_scores = probabilities + self.beta if self.arm == "G4" else logits
                indices = stable_topk(selection_scores, k)
                if self.training and self.arm in ("G2-0.001", "G4"):
                    selected = indices[valid]
                    if len(selected) == 0:
                        raise ValueError("No valid tokens in a training layer")
                    counts = torch.bincount(selected.reshape(-1), minlength=self.E)
                    if self.arm == "G2-0.001":
                        fractions = counts.float() / (len(selected) * k)
                        mean_probability = probabilities[valid].mean(0)
                        unweighted = self.E * (fractions * mean_probability).sum()
                        self.aux = self.aux_weight * unweighted
                        self.last_stats = {
                            "valid_tokens": int(len(selected)), "assignments": int(counts.sum()),
                            "fraction_sum": float(fractions.sum().detach()),
                            "probability_sum": float(mean_probability.sum().detach()),
                            "unweighted_aux": float(unweighted.detach()),
                            "weighted_aux": float(self.aux.detach()),
                        }
                    else:
                        self.pending.add_(counts)
                        self.last_stats = {
                            "valid_tokens": int(len(selected)), "assignments": int(counts.sum()),
                            "counts": counts.detach().cpu().tolist(),
                        }

            weights = k * F.softmax(logits.gather(-1, indices), dim=-1)
            return indices, weights

    @torch.no_grad()
    def after_step(self):
        self.last_update = None
        if self.arm == "G4" and self.training and self.pending.sum() > 0:
            counts = self.pending.float()
            before = self.beta.clone()
            self.beta.add_(
                self.spec["bias_update_rate"] * torch.sign(counts.mean() - counts)
            )
            self.last_update = {
                "counts": counts.long().cpu().tolist(),
                "assignments": int(counts.sum()),
                "beta_before_min": float(before.min()), "beta_before_max": float(before.max()),
                "beta_after_min": float(self.beta.min()), "beta_after_max": float(self.beta.max()),
                "pending_counts_after_update": 0,
            }
        self.pending.zero_()

