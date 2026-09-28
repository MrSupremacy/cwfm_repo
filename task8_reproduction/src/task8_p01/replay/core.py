from __future__ import annotations

import torch

from task8_p01.routing.interventions import route


def replay_ffn(hidden, wi_weight, wo_weight, labels, c0, learned, k, routing):
    """Replay one frozen ReLU FFN on fixed hidden states for all P1 cells.

    Returns exact dense and routed outputs. Bias-free T5-small weights are
    expected. Rows may be tokens from any trace source; no layer propagation
    occurs inside this function.
    """
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    activation = torch.relu(flat @ wi_weight.float().T)
    dense = activation @ wo_weight.float().T
    labels = labels.long().to(flat.device)
    result = {"dense": dense}
    for mode in ("M00", "M01", "M10", "M11"):
        indices, weights = route(
            mode,
            flat,
            c0,
            learned,
            k,
            l2_epsilon=routing["l2_epsilon"],
            rms_epsilon=routing["rms_epsilon"],
            temperature=routing["temperature"],
        )
        coefficients = activation.new_zeros((len(flat), len(c0)))
        coefficients.scatter_(1, indices, weights.to(activation.dtype))
        routed = (activation * coefficients.index_select(1, labels)) @ wo_weight.float().T
        error = torch.linalg.vector_norm(routed - dense, dim=-1) / torch.linalg.vector_norm(
            dense, dim=-1
        ).clamp_min(1.0e-12)
        result[mode] = {
            "indices": indices,
            "weights": weights,
            "output": routed,
            "relative_l2_error": error,
        }
    return result

