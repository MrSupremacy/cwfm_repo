from __future__ import annotations

from task6_phased.common.config import TASKS, dense_run_path, protocol_id
from task6_phased.common.io import terminal_log, write_json
from task6_phased.common.provenance import environment, path_identity
from task6_phased.common.randomness import seed_dropout
from task6_phased.data.datasets import MixedBatchStream, load_raw, move_model_batch, tokenize
from task6_phased.dense.checkpoints import load_dense_checkpoint, save_dense_checkpoint
from task6_phased.dense.model import load_t5, resolve_asset, sample_token_domain_loss
from task6_phased.training.schedule import scheduler_scale


def batch_smoke_dense(config):
    """Run one formal-size forward/backward without an optimizer step or checkpoint."""
    import torch
    source = resolve_asset(config, "pretrained")
    model, tokenizer = load_t5(config, source, trainable=True)
    encoded = {}
    for task in TASKS:
        train, _ = load_raw(config, task)
        encoded[task] = tokenize(
            config, task, train, tokenizer, "train",
            {"batch_smoke": True, "count": len(train)},
        )
    block = config["dense"]
    stream = MixedBatchStream(encoded, tokenizer, block["domain_batch_size"], block["seed"])
    if len(TASKS) * int(block["domain_batch_size"]) != int(block["batch_size"]):
        raise ValueError("Dense formal batch must be four equal domain blocks")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    batch = stream.next()
    domain_ids = batch["domain_id"].to(config["execution"]["device"])
    values = move_model_batch(batch, config["execution"]["device"])
    output = model(**values, use_cache=False)
    loss, domain_losses, _ = sample_token_domain_loss(output.logits, values["labels"], domain_ids)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Dense-MT batch-smoke loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        block["max_grad_norm"], norm_type=2.0, error_if_nonfinite=True, foreach=False,
    )
    result = {
        "status": "passed", "optimizer_step_performed": False,
        "batch_size": int(values["input_ids"].shape[0]),
        "source_tokens": int(values["attention_mask"].sum()),
        "target_tokens": int((values["labels"] != -100).sum()),
        "loss": float(loss.detach()),
        "domain_losses": {
            task: float(domain_losses[index].detach()) for index, task in enumerate(TASKS)
        },
        "grad_norm_before_clip": float(grad_norm),
    }
    if torch.cuda.is_available():
        result.update({
            "device_name": torch.cuda.get_device_name(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        })
    return result


def train_dense(config, run_id, resume=None):
    import torch
    source = resolve_asset(config, "pretrained")
    directory = dense_run_path(config, "train", run_id)
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("Resume requires the existing Dense run directory")
    with terminal_log(directory / "logs/train.log"):
        model, tokenizer = load_t5(config, source, trainable=True)
        source_identity = path_identity(source)
        encoded = {}
        data_identity = {}
        for task in TASKS:
            train, _ = load_raw(config, task)
            data_identity[task] = {"fingerprint": train._fingerprint, "count": len(train)}
            encoded[task] = tokenize(config, task, train, tokenizer, "train", data_identity[task])
        block = config["dense"]
        stream = MixedBatchStream(encoded, tokenizer, block["domain_batch_size"], block["seed"])
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters or sum(p.numel() for p in parameters) != sum(p.numel() for p in model.parameters()):
            raise ValueError("Dense-MT must train every model parameter exactly once")
        optimizer = torch.optim.Adam(
            parameters, lr=block["lr"], betas=tuple(block["betas"]), eps=block["eps"],
            weight_decay=block["weight_decay"], amsgrad=block["amsgrad"],
            foreach=block["foreach"], fused=block["fused"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: scheduler_scale(step, block["total_optimizer_steps"], block["warmup_steps"])
        )
        common = {
            "protocol": protocol_id(config), "run_id": run_id,
            "pretrained_sha256": source_identity, "data": data_identity,
            "total_steps": block["total_optimizer_steps"],
        }
        seed_dropout(block["seed"])
        checkpoints = directory / "checkpoints"
        step = 0
        if resume:
            path = checkpoints / resume
            step = int(load_dense_checkpoint(path, model, optimizer, scheduler, stream, common)["step"])
        else:
            write_json(directory / "config.json", {"config": config, "environment": environment(), **common})
            save_dense_checkpoint(
                checkpoints / "step_0", model, tokenizer, optimizer, scheduler, stream,
                {**common, "name": "step_0", "step": 0, "domain_exposure": dict.fromkeys(TASKS, 0)},
            )
        model.train()
        window = []
        while step < block["total_optimizer_steps"]:
            batch = stream.next()
            domain_ids = batch["domain_id"].to(config["execution"]["device"])
            values = move_model_batch(batch, config["execution"]["device"])
            optimizer.zero_grad(set_to_none=True)
            output = model(**values, use_cache=False)
            loss, domain_losses, _ = sample_token_domain_loss(output.logits, values["labels"], domain_ids)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Dense-MT loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, block["max_grad_norm"], norm_type=2.0,
                error_if_nonfinite=True, foreach=False,
            )
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            row = {
                "step": step, "loss": float(loss.detach()), "domain_losses": {
                    task: float(domain_losses[index].detach()) for index, task in enumerate(TASKS)
                }, "lr": lr_used, "grad_norm": float(norm),
                "domain_exposure": {
                    task: step * block["domain_batch_size"] for task in TASKS
                }, "cycles": dict(stream.sampler.cycles),
            }
            window.append(row)
            if step % block["log_every_steps"] == 0:
                print(row)
                with (directory / "telemetry.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                    import json
                    for item in window:
                        handle.write(json.dumps(item, sort_keys=True) + "\n")
                window.clear()
            if step % block["checkpoint_every_steps"] == 0:
                save_dense_checkpoint(
                    checkpoints / f"step_{step}", model, tokenizer, optimizer, scheduler, stream,
                    {**common, "name": f"step_{step}", "step": step,
                     "domain_exposure": row["domain_exposure"]},
                )
        if window:
            with (directory / "telemetry.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                import json
                for item in window:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
        print("Dense-MT training complete; evaluation and selection are separate stages.")
