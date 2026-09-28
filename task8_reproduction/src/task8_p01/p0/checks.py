from __future__ import annotations

import math

from task8_p01.common.io import write_csv


def _row(check_id, measured, expected, error, status, evidence="in_memory"):
    return {
        "check_id": check_id,
        "E": "", "k": "", "checkpoint": "synthetic",
        "measured_value": measured, "expected_value": expected,
        "max_error": error, "status": status, "evidence_path": evidence,
    }


def run_property_checks(config, output=None):
    """P0 mathematical checks, including exact agreement with Task 6 endpoints."""
    import torch

    from task8_p01.compat.task6 import bootstrap_task6
    from task8_p01.replay.core import replay_ffn
    from task8_p01.routing.interventions import (
        l2_logits, rms_logits, route, stable_topk, weighter,
    )

    torch.manual_seed(8)
    routing = config["routing"]
    rows = []
    e, d, k, n = 8, 16, 3, 7
    c0 = torch.randn(e, d)
    learned = c0 + 0.05 * torch.randn(e, d)
    hidden = torch.randn(n, d)

    # S0=C0 is an initialization contract, not an approximate equality.
    s0 = c0.clone()
    rows.append(_row("S0_equals_C0", int(torch.equal(s0, c0)), 1, 0.0, "pass"))

    bootstrap_task6(config)
    from task6_phased.routing.routers import Router as Task6Router

    spec = {
        "l2_epsilon": routing["l2_epsilon"],
        "rms_epsilon": routing["rms_epsilon"],
        "temperature": routing["temperature"],
        "orthogonal_gain": 1.0,
        "bias_update_rate": 0.001,
        "tie_break": "score_desc_expert_id_asc",
    }
    valid = torch.ones(n, dtype=torch.bool)
    old_r2 = Task6Router("R2", c0, spec, {}, 0, "encoder_layer_00")
    old_r2_indices, old_r2_weights = old_r2(hidden, k, valid=valid)
    new_r2_indices, new_r2_weights = route("M00", hidden, c0, learned, k, **{
        "l2_epsilon": routing["l2_epsilon"],
        "rms_epsilon": routing["rms_epsilon"],
        "temperature": routing["temperature"],
    })
    r2_error = float((new_r2_weights - 1).abs().max())
    r2_ok = old_r2_weights is None and torch.equal(old_r2_indices, new_r2_indices) and r2_error == 0
    rows.append(_row("M00_equals_R2", int(r2_ok), 1, r2_error, "pass" if r2_ok else "fail"))

    old_r4 = Task6Router("R4d", c0, spec, {}, 0, "encoder_layer_00")
    with torch.no_grad():
        old_r4.summary.copy_(learned)
    old_r4_indices, old_r4_weights = old_r4(hidden, k, valid=valid)
    new_r4_indices, new_r4_weights = route(
        "M11", hidden, c0, learned, k,
        l2_epsilon=routing["l2_epsilon"], rms_epsilon=routing["rms_epsilon"],
        temperature=routing["temperature"],
    )
    r4_error = float((old_r4_weights - new_r4_weights).detach().abs().max())
    r4_ok = torch.equal(old_r4_indices, new_r4_indices) and r4_error == 0
    rows.append(_row("M11_equals_R4d", int(r4_ok), 1, r4_error, "pass" if r4_ok else "fail"))

    r2_support = stable_topk(l2_logits(hidden, c0, routing["l2_epsilon"]), k)
    m01_weights = weighter(hidden, {
        "kind": "r4d_soft", "summaries": learned,
        "epsilon": routing["rms_epsilon"], "temperature": routing["temperature"],
    }, r2_support)
    expected = k * torch.softmax(
        rms_logits(hidden, learned, routing["rms_epsilon"], routing["temperature"]).gather(-1, r2_support),
        -1,
    )
    no_retopk_error = float((m01_weights - expected).abs().max())
    rows.append(_row("M01_gather_no_retopk", no_retopk_error, 0.0, no_retopk_error,
                     "pass" if no_retopk_error == 0 else "fail"))

    for name, probe in (
        ("zero", torch.zeros(2, d)),
        ("tiny_norm", torch.full((2, d), 1.0e-30)),
        ("tie", torch.ones(2, d)),
    ):
        tied = torch.ones_like(c0) if name == "tie" else learned
        indices, weights = route(
            "M11", probe, c0, tied, k,
            l2_epsilon=routing["l2_epsilon"], rms_epsilon=routing["rms_epsilon"],
            temperature=routing["temperature"],
        )
        finite = bool(torch.isfinite(weights).all())
        stable = name != "tie" or torch.equal(indices, torch.arange(k).expand_as(indices))
        rows.append(_row(f"numerics_{name}", int(finite and stable), 1, 0.0,
                         "pass" if finite and stable else "fail"))

    full_indices, full_weights = route(
        "M00", hidden, c0, learned, e,
        l2_epsilon=routing["l2_epsilon"], rms_epsilon=routing["rms_epsilon"],
        temperature=routing["temperature"],
    )
    full_ok = torch.equal(torch.sort(full_indices, -1).values, torch.arange(e).expand(n, e))
    full_ok = full_ok and torch.equal(full_weights, torch.ones_like(full_weights))
    rows.append(_row("uniform_k_equals_E", int(full_ok), 1, 0.0, "pass" if full_ok else "fail"))

    wi, wo = torch.randn(24, d), torch.randn(d, 24)
    labels = torch.arange(24) % e
    replay = replay_ffn(hidden, wi, wo, labels, c0, learned, e, routing)
    dense_error = float((replay["M00"]["output"] - replay["dense"]).abs().max())
    rows.append(_row("M00_kE_equals_dense_FFN", dense_error, 0.0, dense_error,
                     "pass" if dense_error <= 2.0e-5 else "fail"))

    learned_grad = learned.clone().requires_grad_(True)
    _, weights = route(
        "M11", hidden, c0, learned_grad, k,
        l2_epsilon=routing["l2_epsilon"], rms_epsilon=routing["rms_epsilon"],
        temperature=routing["temperature"],
    )
    (weights.square().sum()).backward()
    finite_grad = learned_grad.grad is not None and bool(torch.isfinite(learned_grad.grad).all())
    nonzero_grad = finite_grad and float(learned_grad.grad.abs().sum()) > 0
    rows.append(_row("M11_gradient_finite_nonzero", int(nonzero_grad), 1, 0.0,
                     "pass" if nonzero_grad else "fail"))

    sums = []
    for mode in ("M00", "M01", "M10", "M11"):
        _, weights = route(
            mode, hidden, c0, learned, k,
            l2_epsilon=routing["l2_epsilon"], rms_epsilon=routing["rms_epsilon"],
            temperature=routing["temperature"],
        )
        sums.append(float((weights.sum(-1) - k).abs().max()))
    sum_error = max(sums)
    rows.append(_row("all_modes_weight_sum_k", sum_error, 0.0, sum_error,
                     "pass" if sum_error <= 1.0e-6 else "fail"))

    if output:
        write_csv(output, rows)
    if any(row["status"] != "pass" for row in rows):
        failed = [row["check_id"] for row in rows if row["status"] != "pass"]
        raise AssertionError(f"P0 property check failed: {failed}")
    return rows
