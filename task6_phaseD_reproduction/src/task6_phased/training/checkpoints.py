from __future__ import annotations

import os
from pathlib import Path
import uuid

from task6_phased.common.config import protocol_id, run_path
from task6_phased.common.io import checked_complete, complete, fresh_output, read_json, write_json
from task6_phased.common.randomness import restore_rng, rng_state


def save_checkpoint(path, controller, optimizer, scheduler, stream, meta):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    if any(wrapper.router.pending.sum().item() for wrapper in controller.wrappers.values()):
        raise RuntimeError("Cannot checkpoint pending G4 counts")
    staging = path.with_name(f".incomplete_{path.name}_{uuid.uuid4().hex}")
    with fresh_output(staging):
        torch.save({
            "routers": controller.state(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "sampler": stream.state_dict(),
            "rng": rng_state(),
        }, staging / "state.pt")
        write_json(staging / "meta.json", meta)
        complete(staging, meta)
    os.rename(staging, path)


def restore_checkpoint(path, controller, optimizer=None, scheduler=None, stream=None, expected=None):
    import torch
    meta = checked_complete(path)
    for key, value in (expected or {}).items():
        if meta.get(key) != value:
            raise ValueError(f"Router checkpoint {key} mismatch")
    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    controller.load_state(state["routers"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stream.load_state_dict(state["sampler"])
        restore_rng(state["rng"])
    return meta


def states(config, condition, run_id):
    if not condition.trainable:
        return [{"name": "static", "step": 0, "condition": condition.to_dict(), "protocol": protocol_id(config)}]
    checkpoints = run_path(config, "train", condition, run_id) / "checkpoints"
    total = config["training"]["total_optimizer_steps"]
    every = config["training"]["checkpoint_every_steps"]
    result = []
    for step in range(0, total + 1, every):
        path = checkpoints / f"step_{step}"
        meta = checked_complete(path)
        if meta["condition"] != condition.to_dict() or meta["protocol"] != protocol_id(config) or meta["step"] != step:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        result.append(meta)
    return result


def load_for_capture(config, condition, run_id, state, controller, header):
    if condition.trainable:
        restore_checkpoint(
            run_path(config, "train", condition, run_id) / "checkpoints" / state["name"],
            controller, expected={"condition": condition.to_dict(), "protocol": protocol_id(config), "input_header": header},
        )
