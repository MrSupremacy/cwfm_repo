from __future__ import annotations

import math
import torch

from task5.common.config import protocol_id, run_path
from task5.common.context import environment, load_context
from task5.common.io import read_json, terminal_log, write_json
from task5.common.randomness import seed_dropout
from task5.data.datasets import make_loader, move_batch
from task5.training.checkpoints import restore_checkpoint, save_checkpoint
from task5.training.schedule import scheduler_scale


def train_condition(config, condition, run_id, resume=None, stop_after_epoch=None):
    if not condition.trainable:
        raise ValueError("Only trainable router arms may enter training; backbone remains frozen")
    if resume is not None and resume != "final" and not (resume.startswith("step_") and resume[5:].isdigit()):
        raise ValueError("Resume names are 'final' or 'step_<number>', not arbitrary paths")
    directory = run_path(config, "train", condition, run_id)
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("Resume requires the same existing run directory")
    with terminal_log(directory / "logs/train.log"):
        model, tokenizer, controller, data, header, _ = load_context(config, condition, run_id, training=True)
        t = config["training"]
        shuffle_rng = torch.Generator().manual_seed(condition.seed)
        loader = make_loader(config, data, tokenizer, t["batch_size"], shuffle_rng)
        total = len(loader) * t["epochs"]
        warmup = math.ceil(t["warmup_ratio"] * total)
        parameters = [p for p in model.parameters() if p.requires_grad]
        expected = sum(p.numel() for w in controller.wrappers.values() for p in w.router.parameters() if p.requires_grad)
        if not parameters or sum(p.numel() for p in parameters) != expected:
            raise ValueError("Trainable parameters must be exactly router parameters")
        optimizer = torch.optim.Adam(parameters, lr=t["lr"], betas=tuple(t["betas"]), eps=t["eps"],
                                     weight_decay=t["weight_decay"], amsgrad=t["amsgrad"], foreach=t["foreach"], fused=t["fused"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: scheduler_scale(step, total, warmup))
        common = {"condition": condition.to_dict(), "protocol": protocol_id(config), "input_header": header,
                  "total_steps": total, "warmup_steps": warmup}
        seed_dropout(condition.seed)
        epoch, step = 0, 0
        checkpoints = directory / "checkpoints"
        if resume is not None:
            path = checkpoints / resume
            meta = restore_checkpoint(path, controller, optimizer, scheduler, shuffle_rng, common)
            epoch, step = meta["epoch"], meta["step"]
            for other in checkpoints.iterdir():
                if other.is_dir() and not other.name.startswith(".") and read_json(other / "meta.json")["epoch"] > epoch:
                    raise ValueError("Cannot resume behind existing later checkpoints; preserve them and choose another run")
            print(f"RESUME epoch={epoch} step={step}")
        else:
            write_json(directory / "config.json", {"config": config, "environment": environment(), "inputs": header,
                                                   "condition": condition.to_dict(), "trainable_parameters": expected})
            save_checkpoint(checkpoints / "step_0", controller, optimizer, scheduler, shuffle_rng,
                            dict(common, name="step_0", epoch=0, step=0))
        model.train()
        stop = t["epochs"] if stop_after_epoch is None else stop_after_epoch
        if not epoch <= stop <= t["epochs"]:
            raise ValueError("Invalid stop_after_epoch")
        window_norm, window_clips, window_count = 0.0, 0, 0
        for epoch_index in range(epoch, stop):
            for batch in loader:
                batch = move_batch(batch, config["execution"]["device"])
                controller.teacher_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                task_loss = model(**batch, use_cache=False).loss
                aux = controller.aux_loss()
                loss = task_loss + aux
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, t["max_grad_norm"], norm_type=2.0, error_if_nonfinite=True, foreach=False)
                lr_used = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                controller.after_step()
                step += 1
                window_norm += norm.item()
                window_clips += int(norm.item() > t["max_grad_norm"])
                window_count += 1
                if step % config["tasks"][condition.task]["log_every_steps"] == 0 or step % len(loader) == 0:
                    print(f"epoch={epoch_index+1} step={step}/{total} task={task_loss.item():.8g} aux={aux.item():.8g} "
                          f"total={loss.item():.8g} lr_used={lr_used:.8g} grad_norm={norm.item():.8g} "
                          f"mean_grad_norm={window_norm/window_count:.8g} clip_fraction={window_clips/window_count:.6f}")
                    window_norm, window_clips, window_count = 0.0, 0, 0
            finished = epoch_index + 1
            name = "final" if finished == t["epochs"] else f"step_{step}"
            save_checkpoint(checkpoints / name, controller, optimizer, scheduler, shuffle_rng,
                            dict(common, name=name, epoch=finished, step=step))
        print(f"Saved through epoch={stop}; no validation or probe forward executed by training.")
