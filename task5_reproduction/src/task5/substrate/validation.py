from __future__ import annotations

import copy
import numpy as np

from task5.common.config import Condition, conditions, root_for
from task5.common.context import load_context
from task5.common.io import fresh_output, terminal_log, write_json
from task5.data.datasets import make_loader, move_batch


def validate_task(config, task, run_id):
    import torch
    from task5.substrate.model import attach, ffn_layers, load_dense
    from task5.substrate.assets import inspect_task
    from task5.routing.routers import raw_centroids, Router
    path = root_for(config) / "runs/validate" / task / run_id
    with fresh_output(path), terminal_log(path / "logs/validate.log"), torch.no_grad():
        reference, tokenizer, _, data, _, _ = load_context(config, Condition(task, "dense"), run_id)
        batch = next(iter(make_loader(config, data.select(range(2)), tokenizer, 2)))
        x = move_batch(batch, config["execution"]["device"])
        reference.eval()
        baseline = reference(**x, use_cache=False).logits.detach().cpu()
        results = []
        validation_config = copy.deepcopy(config)
        validation_config["suite"]["top_k"] = [6, 13, 19, 26]
        selected_conditions = [c for c in conditions(validation_config) if c.task == task and c.arm != "dense" and c.seed in (0, None)]
        for condition in selected_conditions:
            model, _, ctrl, _, _, _ = load_context(config, condition, run_id)
            model.eval()
            for w in ctrl.wrappers.values():
                w.force_all = True
            ctrl.teacher_batch(x)
            output = model(**x, use_cache=False).logits.detach().cpu()
            error = float((output-baseline).abs().max())
            if error >= 1e-5:
                raise AssertionError(f"Phase0 failed: {condition}, max_abs={error}")
            layer_errors = {}
            for key, w in ctrl.wrappers.items():
                generator = torch.Generator().manual_seed(0)
                hidden = torch.randn(1, 256, config["model"]["d_model"], generator=generator).to(x["input_ids"].device)
                w.valid = w.token_ids = None
                expected = w.wo(w.dropout(w.act(w.wi(hidden))))
                layer_error = float((w(hidden)-expected).abs().max())
                if layer_error >= 1e-5:
                    raise AssertionError("Layer full-selection equivalence failed")
                layer_errors[key] = layer_error
                w.force_all = False
            ctrl.teacher_batch(x)
            normal = model(**x, use_cache=False).logits
            if not torch.isfinite(normal).all():
                raise AssertionError("Non-finite normal-route output")
            results.append({"condition": condition.to_dict(), "force_all_max_abs": error, "layers": layer_errors})
            del model, ctrl
        # Same-input R2/R3 diagnostic only: the user's RMS decision overrides the old equality gate.
        assets, labels, _ = inspect_task(config, task)
        model, _ = load_dense(config, assets["dense"])
        agreement = {}
        for key, _, _, module in ffn_layers(model):
            cents = raw_centroids(module.wi.weight, torch.as_tensor(labels[key], device=module.wi.weight.device), config["model"]["num_experts"])
            r2 = Router("R2", cents, config["routing"], {}, None, key).eval()
            r3 = Router("R3", cents, config["routing"], {}, None, key).eval()
            r4_r2init = Router("R4-R2Init", cents, config["routing"], {}, 0, key).eval()
            if not torch.equal(r4_r2init.summary.detach(), cents) or not r4_r2init.summary.requires_grad:
                raise AssertionError("R4-R2Init must start from the exact shared centroid as a trainable parameter")
            hidden = torch.randn(256, config["model"]["d_model"], generator=torch.Generator().manual_seed(0)).to(cents.device)
            values = {}
            for k in config["suite"]["top_k"]:
                valid = torch.ones(len(hidden), dtype=torch.bool, device=hidden.device)
                a, _ = r2(hidden, k, q=None, valid=valid)
                b, _ = r3(hidden, k, q=None, valid=valid)
                c, weights = r4_r2init(hidden, k, q=None, valid=valid)
                if not torch.equal(b, c):
                    raise AssertionError("R4-R2Init and R3 must select identical experts at step 0")
                if not torch.allclose(weights.sum(-1), torch.full((len(hidden),), float(k), device=hidden.device)):
                    raise AssertionError("R4-R2Init soft weights must sum to k")
                shared = (a[:, :, None] == b[:, None, :]).any(-1).float().sum(-1) / k
                values[str(k)] = {"overlap": shared.mean().item(), "exact_set_agreement": (shared == 1).float().mean().item(),
                                  "r4_r2init_r3_exact_set_agreement": float((b == c).all(-1).float().mean())}
            agreement[key] = values
        write_json(path / "phase0.json", {"results": results, "r2_r3_diagnostic_only": agreement,
                                          "r2_r3_equal_sets_required": False, "tf32": False})
        print(f"Phase0 passed for {task}; R2/R3 agreement recorded without an equality threshold.")
