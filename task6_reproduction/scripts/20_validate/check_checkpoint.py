#!/usr/bin/env python
"""Strictly restore one complete fullFT checkpoint without taking a training step."""
from __future__ import annotations

import argparse
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/suites/smoke.yaml")
    parser.add_argument("--local", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True, choices=("sst2", "mnli"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--state", default="step_1")
    args = parser.parse_args()

    import torch
    from task6.common.config import Condition, load_config, protocol_id, run_path
    from task6.common.context import load_context
    from task6.common.randomness import configure_torch
    from task6.data.datasets import make_loader, move_batch
    from task6.training.checkpoints import restore_checkpoint
    from task6.training.parameters import parameter_groups
    from task6.training.schedule import scheduler_scale

    config = load_config(args.suite, args.local)
    configure_torch(config)
    condition = Condition(args.task, args.arm, args.variant, args.k, args.seed)
    model, tokenizer, controller, data, header, _ = load_context(
        config, condition, args.run_id, training=True)
    training = config["training"]
    shuffle = torch.Generator().manual_seed(args.seed)
    loader = make_loader(config, data, tokenizer, training["batch_size"], shuffle)
    updates = math.ceil(len(loader) / training["accumulation_steps"])
    total = updates * training["epochs"]
    warmup = math.ceil(training["warmup_ratio"] * total)
    groups, manifest = parameter_groups(model, controller, condition, training)
    optimizer = torch.optim.Adam(
        groups, betas=tuple(training["betas"]), eps=training["eps"],
        weight_decay=training["weight_decay"], amsgrad=training["amsgrad"],
        foreach=training["foreach"], fused=training["fused"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scheduler_scale(step, total, warmup))
    path = run_path(config, "train", condition, args.run_id) / "checkpoints" / args.state
    meta = restore_checkpoint(path, model, controller, optimizer, scheduler, shuffle, {
        "condition": condition.to_dict(), "protocol": protocol_id(config),
        "input_header": header, "total_steps": total, "warmup_steps": warmup,
        "parameter_manifest": manifest,
    })
    if not optimizer.state or scheduler.state_dict()["last_epoch"] != meta["step"]:
        raise AssertionError("Optimizer or scheduler state did not round-trip")
    batch = next(iter(make_loader(config, data.select(range(2)), tokenizer, 2)))
    values = move_batch(batch, config["execution"]["device"])
    model.eval()
    controller.teacher_batch(values)
    with torch.no_grad():
        logits = model(**values, use_cache=False).logits
    if not torch.isfinite(logits).all():
        raise AssertionError("Restored model produced non-finite logits")
    print({"checkpoint": str(path), "epoch": meta["epoch"], "step": meta["step"],
           "model_tensors": len(model.state_dict()), "optimizer_states": len(optimizer.state),
           "parameter_groups": [group["name"] for group in groups]})


if __name__ == "__main__":
    main()
