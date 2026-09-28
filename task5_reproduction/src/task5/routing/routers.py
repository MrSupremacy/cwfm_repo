from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from task5.common.randomness import stream_seed


def stable_topk(scores, k):
    if not 1 <= k <= scores.shape[-1] or not torch.isfinite(scores).all():
        raise ValueError("Invalid top-k scores/budget")
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


def raw_centroids(wi, labels, experts):
    return torch.stack([wi.detach().float()[labels == e].mean(0) for e in range(experts)])


class Router(nn.Module):
    """One router per layer; no ownership of frozen FFN weights."""

    def __init__(self, arm, centroids, spec, variant, seed, layer, hash_table=None):
        super().__init__()
        self.arm, self.spec = arm, dict(spec)
        self.E, self.D = centroids.shape
        self.aux_weight = float(variant.get("aux_weight", 0.0))
        self.noise_rng = None
        self.noise_seed = stream_seed(seed or 0, "noise", layer)
        self.aux = None
        self.register_buffer("pending", torch.zeros(self.E, dtype=torch.long), persistent=False)
        if arm in ("R2", "R2-soft"):
            self.register_buffer("centroids", centroids.clone())
        elif arm in ("R3", "R4", "R4-R2Init", "R4-hard"):
            initial = centroids.clone()
            if arm in ("R4", "R4-hard"):
                generator = torch.Generator(device=initial.device).manual_seed(stream_seed(seed, "summary_init", layer))
                nn.init.orthogonal_(initial, gain=spec["orthogonal_gain"], generator=generator)
            self.summary = nn.Parameter(initial, requires_grad=arm != "R3")
        elif arm == "G0":
            if hash_table is None:
                raise ValueError("G0 requires a shared vocabulary permutation table")
            self.register_buffer("table", hash_table)
        elif arm.startswith("G"):
            # Explicit nn.Linear default distribution without consuming the global RNG.
            generator = torch.Generator(device=centroids.device).manual_seed(stream_seed(seed, "clean_gate_init", layer))
            bound = 1 / math.sqrt(self.D)
            self.weight = nn.Parameter(torch.empty_like(centroids).uniform_(-bound, bound, generator=generator))
            self.gate_bias = nn.Parameter(torch.empty(self.E, device=centroids.device).uniform_(-bound, bound, generator=generator))
            if arm == "G3":
                self.noise_weight = nn.Parameter(torch.zeros_like(centroids))
            if arm == "G4":
                self.register_buffer("beta", torch.zeros(self.E, device=centroids.device))

    def forward(self, x, k, *, q, valid, token_ids=None):
        self.aux = None
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.arm == "R1":
                return stable_topk(q, k), None
            if self.arm in ("R2", "R2-soft"):
                eps = self.spec["l2_epsilon"]
                scores = F.normalize(x, dim=-1, eps=eps) @ F.normalize(self.centroids.float(), dim=-1, eps=eps).T
                indices = stable_topk(scores, k)
                # R2 cosine scale, temperature 1: no RMS or sqrt(D) rescaling.
                weights = k * F.softmax(scores.gather(-1, indices), dim=-1) if self.arm == "R2-soft" else None
                return indices, weights
            if self.arm in ("R3", "R4", "R4-R2Init", "R4-hard"):
                eps = self.spec["rms_epsilon"]
                summary = self.summary.float()
                xn = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                sn = summary * torch.rsqrt(summary.square().mean(-1, keepdim=True) + eps)
                logits = (xn @ sn.T) / math.sqrt(self.D) / self.spec["temperature"]
                indices = stable_topk(logits, k)
                if self.arm == "R3":
                    return indices, None
            elif self.arm == "G0":
                if token_ids is None or token_ids.shape != x.shape[:-1]:
                    raise ValueError("G0 needs current input token IDs, including decoder generation steps")
                if (token_ids < 0).any() or (token_ids >= self.table.shape[0]).any():
                    raise ValueError("G0 token ID outside vocabulary")
                return self.table[token_ids, :k].long(), None
            else:
                logits = F.linear(x, self.weight.float(), self.gate_bias.float())
                if self.arm == "G3" and self.training:
                    if self.noise_rng is None:
                        self.noise_rng = torch.Generator(device=x.device).manual_seed(self.noise_seed)
                    sigma = F.softplus(F.linear(x, self.noise_weight.float())) + self.spec["noise_epsilon"]
                    logits = logits + torch.randn(logits.shape, device=x.device, dtype=torch.float32, generator=self.noise_rng) * sigma
                probs = F.softmax(logits, dim=-1)
                scores = probs + self.beta if self.arm == "G4" else logits
                indices = stable_topk(scores, k)
                if self.training and self.arm in ("G2", "G4"):
                    selected = indices[valid]
                    if len(selected) == 0:
                        raise ValueError("No valid tokens in a training layer")
                    counts = torch.bincount(selected.reshape(-1), minlength=self.E)
                    if self.arm == "G2":
                        f = counts.float() / (len(selected) * k)
                        self.aux = self.aux_weight * self.E * (f * probs[valid].mean(0)).sum()
                    else:
                        self.pending.add_(counts)
            weights = k * F.softmax(logits.gather(-1, indices), dim=-1)
            if self.arm == "R4-hard":
                # Coefficient-level straight-through: exactly one forward,
                # soft coefficient derivative backward. Do NOT detach x or
                # expert activations: upstream routers still need that path.
                weights = torch.ones_like(weights) + (weights - weights.detach())
            return indices, weights

    @torch.no_grad()
    def after_step(self):
        if self.arm == "G4" and self.training and self.pending.sum() > 0:
            counts = self.pending.float()
            self.beta.add_(self.spec["bias_update_rate"] * torch.sign(counts.mean() - counts))
        self.pending.zero_()

    def noise_state(self):
        return None if self.noise_rng is None else self.noise_rng.get_state()

    def load_noise_state(self, state):
        if state is not None:
            self.noise_rng = torch.Generator(device=next(self.parameters()).device)
            self.noise_rng.set_state(state.cpu())


def make_hash_table(vocabulary, experts, seed):
    rng = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(experts, generator=rng) for _ in range(vocabulary)]).to(torch.uint8)
