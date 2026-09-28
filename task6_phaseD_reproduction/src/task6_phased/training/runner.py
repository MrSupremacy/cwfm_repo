from __future__ import annotations

import json

from task6_phased.common.config import TASKS, protocol_id, run_path
from task6_phased.common.context import load_context
from task6_phased.common.io import terminal_log, write_json
from task6_phased.common.provenance import environment
from task6_phased.common.randomness import seed_dropout
from task6_phased.data.datasets import MixedBatchStream, move_model_batch
from task6_phased.dense.model import sample_token_domain_loss
from task6_phased.training.checkpoints import restore_checkpoint, save_checkpoint
from task6_phased.training.schedule import scheduler_scale


def train_condition(config, condition, run_id, resume=None):
    import torch
    if not condition.trainable:
        raise ValueError("R2 is static and cannot enter training")
    directory = run_path(config, "train", condition, run_id)
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("Resume requires the existing condition directory")
    with terminal_log(directory / "logs/train.log"):
        model, tokenizer, controller, datasets, header = load_context(
            config, condition, run_id, populations=("train",)
        )
        block = config["training"]
        stream = MixedBatchStream(
            {task: datasets[task]["train"] for task in TASKS}, tokenizer,
            block["domain_batch_size"], condition.seed,
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        expected_parameters = [
            parameter for wrapper in controller.wrappers.values()
            for parameter in wrapper.router.parameters() if parameter.requires_grad
        ]
        if not parameters or {id(p) for p in parameters} != {id(p) for p in expected_parameters}:
            raise ValueError("F0 optimizer membership must equal router parameters exactly")
        optimizer = torch.optim.Adam(
            parameters, lr=block["lr"], betas=tuple(block["betas"]), eps=block["eps"],
            weight_decay=block["weight_decay"], amsgrad=block["amsgrad"],
            foreach=block["foreach"], fused=block["fused"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: scheduler_scale(step, block["total_optimizer_steps"], block["warmup_steps"])
        )
        common = {
            "protocol": protocol_id(config), "condition": condition.to_dict(),
            "input_header": header, "total_steps": block["total_optimizer_steps"],
        }
        seed_dropout(condition.seed)
        checkpoints = directory / "checkpoints"
        step = 0
        if resume:
            step = int(restore_checkpoint(
                checkpoints / resume, controller, optimizer, scheduler, stream, common
            )["step"])
        else:
            write_json(directory / "config.json", {
                "config": config, "condition": condition.to_dict(), "environment": environment(),
                "input_header": header, "trainable_parameters": sum(p.numel() for p in parameters),
            })
            save_checkpoint(
                checkpoints / "step_0", controller, optimizer, scheduler, stream,
                {**common, "name": "step_0", "step": 0, "domain_exposure": dict.fromkeys(TASKS, 0)},
            )
        model.train()
        telemetry = directory / "telemetry.jsonl"
        window = []
        while step < block["total_optimizer_steps"]:
            batch = stream.next()
            domain_ids = batch["domain_id"].to(config["execution"]["device"])
            values = move_model_batch(batch, config["execution"]["device"])
            controller.teacher_batch(values)
            optimizer.zero_grad(set_to_none=True)
            output = model(**values, use_cache=False)
            task_loss, domain_losses, _ = sample_token_domain_loss(output.logits, values["labels"], domain_ids)
            auxiliary = controller.aux_loss()
            loss = task_loss + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite routed loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, block["max_grad_norm"], norm_type=2.0,
                error_if_nonfinite=True, foreach=False,
            )
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            controller.after_step()
            routing_diagnostics = controller.training_diagnostics()
            step += 1
            row = {
                "step": step, "task_loss": float(task_loss.detach()), "aux_loss": float(auxiliary.detach()),
                "total_loss": float(loss.detach()), "domain_losses": {
                    task: float(domain_losses[index].detach()) for index, task in enumerate(TASKS)
                }, "lr": lr_used, "grad_norm": float(norm),
                "domain_exposure": {task: step * block["domain_batch_size"] for task in TASKS},
                "cycles": dict(stream.sampler.cycles),
            }
            if step % block["log_every_steps"] == 0:
                row["routing_diagnostics"] = routing_diagnostics
            window.append(row)
            if step % block["log_every_steps"] == 0:
                print(row)
                with telemetry.open("a", encoding="utf-8", newline="\n") as stream_file:
                    for item in window:
                        stream_file.write(json.dumps(item, sort_keys=True) + "\n")
                window.clear()
            if step % block["checkpoint_every_steps"] == 0:
                save_checkpoint(
                    checkpoints / f"step_{step}", controller, optimizer, scheduler, stream,
                    {**common, "name": f"step_{step}", "step": step, "domain_exposure": row["domain_exposure"]},
                )
        if window:
            with telemetry.open("a", encoding="utf-8", newline="\n") as stream_file:
                for item in window:
                    stream_file.write(json.dumps(item, sort_keys=True) + "\n")
        print("Routed condition complete; capture is a separate stage.")
