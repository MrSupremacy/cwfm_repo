from __future__ import annotations

from task6.common.config import Condition, conditions, protocol_id, root_for
from task6.common.context import load_context
from task6.common.io import fresh_output, terminal_log, write_json
from task6.data.datasets import make_loader, move_batch
from task6.training.parameters import parameter_groups


def _nonzero_finite(parameters):
    import torch

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(value).all() for value in gradients) and any(
        torch.count_nonzero(value).item() for value in gradients)


def validate_task(config, task, run_id):
    """Real-input Phase 0: split equivalence plus fullFT gradient-path checks."""
    import torch

    path = root_for(config) / "runs/validate" / task / run_id
    with fresh_output(path), terminal_log(path / "logs/validate.log"):
        reference, tokenizer, _, data, header, _ = load_context(
            config, Condition(task, "dense"), run_id)
        batch = next(iter(make_loader(config, data.select(range(2)), tokenizer, 2)))
        values = move_batch(batch, config["execution"]["device"])
        reference.eval()
        with torch.no_grad():
            baseline = reference(**values, use_cache=False).logits.detach().cpu()
        del reference

        first_k = config["suite"]["top_k"][0]
        selected = [condition for condition in conditions(config)
                    if condition.task == task and condition.is_routed
                    and condition.k == first_k and condition.seed == config["suite"]["seeds"][0]]
        if len(selected) != len(config["variants"]):
            raise ValueError("Phase 0 did not select exactly one condition per arm")

        results = []
        for condition in selected:
            model, _, controller, _, current_header, _ = load_context(config, condition, run_id)
            if current_header != header:
                raise ValueError("Validation conditions do not share prepared inputs")
            model.eval()
            for wrapper in controller.wrappers.values():
                wrapper.force_all = True
            controller.teacher_batch(values)
            with torch.no_grad():
                output = model(**values, use_cache=False).logits.detach().cpu()
            error = float((output - baseline).abs().max())
            if error >= 1.0e-5:
                raise AssertionError(f"Full-selection equivalence failed: {condition}, max_abs={error}")

            for wrapper in controller.wrappers.values():
                wrapper.force_all = False
            model.train()
            model.zero_grad(set_to_none=True)
            controller.teacher_batch(values)
            task_loss = model(**values, use_cache=False).loss
            aux = sum(controller.aux_losses().values(), task_loss.new_zeros(()))
            loss = task_loss + aux
            loss.backward()
            groups, manifest = parameter_groups(model, controller, condition, config["training"])
            grouped = {group["name"]: group["params"] for group in groups}
            if not _nonzero_finite(grouped["backbone"]):
                raise AssertionError(f"Backbone gradient path failed: {condition}")
            if condition.has_router_parameters and not _nonzero_finite(grouped["router"]):
                raise AssertionError(f"Router gradient path failed: {condition}")
            controller.clear_pending()
            results.append({
                "condition": condition.to_dict(), "force_all_max_abs": error,
                "task_loss": float(task_loss.detach()), "aux_loss": float(aux.detach()),
                "backbone_gradient": True,
                "router_gradient": condition.has_router_parameters,
                "parameter_manifest": manifest,
            })
            del model, controller
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_json(path / "phase0.json", {
            "protocol": protocol_id(config),
            "task": task, "input_header": header, "k_checked": first_k,
            "results": results, "force_all_tolerance": 1.0e-5,
            "tf32": False, "fullft_gradient_paths_checked": True,
        })
        print(f"Phase 0 passed for {task}: eight arms, split equivalence, and fullFT gradients")
