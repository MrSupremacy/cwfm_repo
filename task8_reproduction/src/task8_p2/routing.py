from __future__ import annotations

import numpy as np


def router_arrays(hidden, c0, learned, k, routing, device):
    import torch
    from task8_p01.routing.interventions import l2_logits, rms_logits, stable_topk

    x = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    c = torch.as_tensor(c0, dtype=torch.float32, device=device)
    s = torch.as_tensor(learned, dtype=torch.float32, device=device)
    scores2 = l2_logits(x, c, routing["l2_epsilon"])
    actual = rms_logits(x, s, routing["rms_epsilon"], routing["temperature"])
    ids2, ids4 = stable_topk(scores2, k), stable_topk(actual, k)
    # Double-precision geometry avoids zero/tiny norm underflow; actual z remains
    # the original float32 endpoint computation, never a substitute formula.
    xd, sd = x.double(), s.double()
    nx = torch.linalg.vector_norm(xd, dim=-1)
    ns = torch.linalg.vector_norm(sd, dim=-1)
    q = (xd @ sd.T) / (nx[:, None]*ns[None, :]).clamp_min(1e-300)
    eps_d = x.shape[-1]*float(routing["rms_epsilon"])
    eta_x = nx/torch.sqrt(nx.square()+eps_d)
    eta_s = ns/torch.sqrt(ns.square()+eps_d)
    selected = actual.gather(-1, ids4)
    p = torch.softmax(selected, -1)
    return {name: value.detach().cpu().numpy() for name, value in {
        "scores_R2": scores2, "q_St": q, "z_actual": actual,
        "ids_R2": ids2, "ids_R4d": ids4, "selected_scores": selected,
        "p": p, "eta_x": eta_x,
        "eta_St": eta_s[None, :].expand(len(x), -1),
    }.items()}


def temperature_route(hidden, learned, k, routing, multiplier):
    import torch
    from task8_p01.routing.interventions import rms_logits, stable_topk

    z = rms_logits(hidden, learned, routing["rms_epsilon"], routing["temperature"])
    indices = stable_topk(z, k)  # original selector, independent of multiplier
    selected = z.gather(-1, indices)
    if multiplier == "uniform":
        weights = torch.ones_like(selected)
    else:
        if not np.isfinite(float(multiplier)) or float(multiplier) <= 0:
            raise ValueError("multiplier must be finite and positive")
        weights = int(k)*torch.softmax(selected/float(multiplier), -1)
    return indices, weights


def attach_temperature(config, model, labels, c0, learned, k):
    import torch
    from task8_p01.substrate.model import InterventionFFN, InterventionController, ffn_layers

    class TemperatureFFN(InterventionFFN):
        multiplier = 1

        def forward(self, hidden):
            flat = hidden.reshape(-1, hidden.shape[-1])
            indices, weights = temperature_route(flat, self.learned, self.k, self.routing, self.multiplier)
            activation = self.act(self.wi(hidden))
            aflat = activation.reshape(-1, activation.shape[-1])
            coefficients = aflat.new_zeros((len(aflat), len(self.learned)))
            coefficients.scatter_(1, indices, weights.to(aflat.dtype))
            masked = aflat*coefficients.index_select(1, self.labels)
            return self.wo(self.dropout(masked.reshape_as(activation)))

    wrappers = {}
    for key, stack, parent, original in list(ffn_layers(model)):
        wrapper = TemperatureFFN(original, labels[key], c0[key], learned[key], key, stack, k, config["routing"])
        parent.DenseReluDense = wrapper
        wrappers[key] = wrapper
    if len(wrappers) != 12:
        raise ValueError("Temperature intervention requires exactly 12 FFNs")
    controller = InterventionController(model, wrappers)

    def set_multiplier(multiplier):
        for wrapper in wrappers.values():
            wrapper.multiplier = multiplier
    controller.set_multiplier = set_multiplier
    return controller
