from __future__ import annotations

from task6_phaseb.common.config import ARMS, Condition, budgets_for, root_for
from task6_phaseb.common.context import load_context
from task6_phaseb.common.io import complete, fresh_output, terminal_log, write_json
from task6_phaseb.data.datasets import make_loader, move_batch


def validate_task(config, task, experts, run_id):
    import torch
    from task6_phaseb.routing.routers import Router
    from task6_phaseb.substrate.assets import inspect_task
    from task6_phaseb.substrate.model import load_dense

    output = root_for(config) / "runs/validate" / task / f"E_{experts}" / run_id
    paths, _, centroids, identity = inspect_task(config, task, experts)
    with fresh_output(output), terminal_log(output / "logs/validate.log"):
        reference, _ = load_dense(config, paths["dense"])
        reference.eval()
        first_k = budgets_for(config, experts)[0]
        base_condition = Condition(task, experts, "R2", "default", first_k, None)
        _, tokenizer, _, data, _, _ = load_context(config, base_condition, run_id)
        batch = next(iter(make_loader(config, data.select(range(min(2, len(data)))), tokenizer, 2)))
        model_batch = move_batch(batch, config["execution"]["device"])
        with torch.no_grad():
            baseline = reference(**model_batch, use_cache=False).logits.detach().cpu()

        results = []
        for arm in ARMS:
            seed = None if arm == "R2" else 0
            condition = Condition(task, experts, arm, "default", first_k, seed)
            model, _, controller, _, _, _ = load_context(config, condition, run_id)
            model.eval()
            for wrapper in controller.wrappers.values():
                wrapper.force_all = True
            controller.teacher_batch(model_batch)
            with torch.no_grad():
                logits = model(**model_batch, use_cache=False).logits.detach().cpu()
            error = float((logits - baseline).abs().max())
            if error >= 1.0e-5:
                raise AssertionError(f"Force-all equivalence failed: {condition}; max_abs={error}")
            for wrapper in controller.wrappers.values():
                wrapper.force_all = False
            controller.teacher_batch(model_batch)
            with torch.no_grad():
                ordinary = model(**model_batch, use_cache=False).logits
            if not torch.isfinite(ordinary).all():
                raise AssertionError(f"Non-finite routed output: {condition}")
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            expected = [
                name for name, parameter in model.named_parameters()
                if ".router." in name and parameter.requires_grad
            ]
            if trainable != expected or (condition.trainable and not trainable) or (not condition.trainable and trainable):
                raise AssertionError(f"Trainable scope is not router-only: {condition}")
            results.append({
                "condition": condition.to_dict(),
                "force_all_max_abs": error,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            })
            del model, controller

        layer = sorted(centroids)[0]
        centre = torch.as_tensor(centroids[layer], device=config["execution"]["device"])
        generator = torch.Generator(device=centre.device).manual_seed(0)
        hidden = torch.randn(32, config["model"]["d_model"], device=centre.device, generator=generator)
        valid = torch.ones(len(hidden), dtype=torch.bool, device=hidden.device)
        checks = {}
        variants = {item["arm"]: item for item in config["variants"]}
        for arm in ("R4d", "G2-0.001", "G4"):
            router = Router(arm, centre, config["routing"], variants[arm], 0, layer).to(centre.device).train()
            if arm == "R4d" and (not torch.equal(router.summary.detach(), centre) or not router.summary.requires_grad):
                raise AssertionError("R4d must start at the exact raw centroid and remain trainable")
            selected, weights = router(hidden, budgets_for(config, experts)[-1], valid=valid)
            loss = weights.square().mean() + (router.aux if router.aux is not None else weights.new_zeros(()))
            loss.backward()
            gradients = [p.grad for p in router.parameters() if p.requires_grad]
            if not gradients or any(
                g is None or not torch.isfinite(g).all() or g.abs().sum().item() == 0
                for g in gradients
            ):
                raise AssertionError(f"Invalid router gradient: {arm}")
            before = router.beta.clone() if arm == "G4" else None
            router.after_step()
            if arm == "G4" and torch.equal(before, router.beta):
                raise AssertionError("G4 beta did not update after a training step")
            if selected.max() >= experts or selected.min() < 0:
                raise AssertionError("Selected expert ID is out of range")
            checks[arm] = {"finite_gradients": True, "selected_max": int(selected.max())}

        header = {"schema": 1, "task": task, "experts": experts, "identity": identity}
        write_json(output / "validation.json", {"header": header, "conditions": results, "high_e_checks": checks})
        complete(output, header)
        print(f"Phase B validation passed: {task}/E{experts}")
