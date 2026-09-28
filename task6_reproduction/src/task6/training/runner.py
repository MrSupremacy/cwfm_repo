from __future__ import annotations

import math

import torch

from task6.common.config import protocol_id, run_path
from task6.common.context import environment, load_context
from task6.common.io import read_json, terminal_log, write_json
from task6.common.randomness import seed_dropout
from task6.data.datasets import make_loader, move_batch
from task6.training.checkpoints import restore_checkpoint, save_checkpoint
from task6.training.parameters import gradient_norm, parameter_groups
from task6.training.schedule import scheduler_scale


def _windows(loader, size):
    window = []
    for batch in loader:
        window.append(batch)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def _token_counts(batch):
    return {
        "task": int((batch["labels"] != -100).sum()),
        "encoder": int(batch["attention_mask"].sum()),
        "decoder": int((batch["labels"] != -100).sum()),
    }


def train_condition(config, condition, run_id, resume=None, stop_after_epoch=None):
    if not condition.trainable:
        raise ValueError("Dense reference cannot enter F1 training")
    if resume is not None and resume != "final" and not (resume.startswith("step_") and resume[5:].isdigit()):
        raise ValueError("Resume names are 'final' or 'step_<epoch>'")
    directory = run_path(config, "train", condition, run_id)
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("Resume requires the same existing run directory")

    with terminal_log(directory / "logs/train.log"):
        model, tokenizer, controller, data, header, _ = load_context(config, condition, run_id, training=True)
        training = config["training"]
        shuffle_rng = torch.Generator().manual_seed(condition.seed)
        loader = make_loader(config, data, tokenizer, training["batch_size"], shuffle_rng)
        updates_per_epoch = math.ceil(len(loader) / training["accumulation_steps"])
        total_steps = updates_per_epoch * training["epochs"]
        warmup_steps = math.ceil(training["warmup_ratio"] * total_steps)
        groups, parameter_manifest = parameter_groups(model, controller, condition, training)
        optimizer = torch.optim.Adam(
            groups, betas=tuple(training["betas"]), eps=training["eps"],
            weight_decay=training["weight_decay"], amsgrad=training["amsgrad"],
            foreach=training["foreach"], fused=training["fused"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: scheduler_scale(step, total_steps, warmup_steps))
        common = {
            "condition": condition.to_dict(), "protocol": protocol_id(config), "input_header": header,
            "total_steps": total_steps, "warmup_steps": warmup_steps,
            "parameter_manifest": parameter_manifest,
        }
        all_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        seed_dropout(condition.seed)
        epoch, step = 0, 0
        checkpoints = directory / "checkpoints"
        if resume is not None:
            meta = restore_checkpoint(
                checkpoints / resume, model, controller, optimizer, scheduler, shuffle_rng, common)
            epoch, step = meta["epoch"], meta["step"]
            for other in checkpoints.iterdir():
                if other.is_dir() and not other.name.startswith(".") and read_json(other / "meta.json")["epoch"] > epoch:
                    raise ValueError("Cannot resume behind a later retained checkpoint")
            print(f"RESUME epoch={epoch} step={step}")
        else:
            write_json(directory / "config.json", {
                "config": config, "environment": environment(), "inputs": header,
                "condition": condition.to_dict(), "parameter_manifest": parameter_manifest,
            })
            save_checkpoint(
                checkpoints / "step_0", model, controller, optimizer, scheduler, shuffle_rng,
                dict(common, name="step_0", epoch=0, step=0),
            )

        model.train()
        stop = training["epochs"] if stop_after_epoch is None else stop_after_epoch
        if not epoch <= stop <= training["epochs"]:
            raise ValueError("Invalid stop_after_epoch")
        window_norm, window_clips, window_count = 0.0, 0, 0

        for epoch_index in range(epoch, stop):
            for batches in _windows(loader, training["accumulation_steps"]):
                counts = [_token_counts(batch) for batch in batches]
                denominators = {key: sum(item[key] for item in counts) for key in ("task", "encoder", "decoder")}
                if any(value <= 0 for value in denominators.values()):
                    raise ValueError("An optimizer window has no effective tokens")
                optimizer.zero_grad(set_to_none=True)
                task_value = aux_value = total_value = 0.0
                for raw_batch, count in zip(batches, counts):
                    batch = move_batch(raw_batch, config["execution"]["device"])
                    controller.teacher_batch(batch)
                    task_loss = model(**batch, use_cache=False).loss
                    aux_by_layer = controller.aux_losses()
                    weighted_task = task_loss * (count["task"] / denominators["task"])
                    weighted_aux = task_loss.new_zeros(())
                    for layer, aux in aux_by_layer.items():
                        stack = controller.wrappers[layer].stack
                        weighted_aux = weighted_aux + aux * (count[stack] / denominators[stack])
                    loss = weighted_task + weighted_aux
                    if not torch.isfinite(loss):
                        controller.clear_pending()
                        raise FloatingPointError("Non-finite fullFT training loss")
                    loss.backward()
                    task_value += float(weighted_task.detach())
                    aux_value += float(weighted_aux.detach())
                    total_value += float(loss.detach())

                group_norms = {group["name"]: gradient_norm(group["params"]) for group in optimizer.param_groups}
                total_norm = torch.nn.utils.clip_grad_norm_(
                    all_parameters, training["max_grad_norm"], norm_type=2.0,
                    error_if_nonfinite=True, foreach=False,
                )
                lrs_used = {group["name"]: group["lr"] for group in optimizer.param_groups}
                optimizer.step()
                scheduler.step()
                controller.after_step()
                step += 1
                window_norm += float(total_norm)
                window_clips += int(float(total_norm) > training["max_grad_norm"])
                window_count += 1
                interval = config["tasks"][condition.task]["log_every_steps"]
                if step % interval == 0 or step % updates_per_epoch == 0:
                    print(
                        f"epoch={epoch_index + 1} step={step}/{total_steps} "
                        f"task={task_value:.8g} aux={aux_value:.8g} total={total_value:.8g} "
                        f"lr_backbone={lrs_used['backbone']:.8g} "
                        f"lr_router={lrs_used.get('router', 0.0):.8g} "
                        f"grad_total={float(total_norm):.8g} "
                        f"grad_backbone={group_norms['backbone']:.8g} "
                        f"grad_router={group_norms.get('router', 0.0):.8g}"
                    )

            completed_epoch = epoch_index + 1
            name = "final" if completed_epoch == training["epochs"] else f"step_{completed_epoch}"
            meta = dict(
                common, name=name, epoch=completed_epoch, step=step,
                mean_preclip_grad_norm=window_norm / max(1, window_count),
                clip_fraction=window_clips / max(1, window_count),
            )
            save_checkpoint(checkpoints / name, model, controller, optimizer, scheduler, shuffle_rng, meta)
            print(f"CHECKPOINT {name} epoch={completed_epoch} step={step}")
            window_norm = 0.0
            window_clips = window_count = 0
