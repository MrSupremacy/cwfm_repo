from __future__ import annotations

from pathlib import Path
import os
import uuid

from task6.common.config import protocol_id, run_path
from task6.common.io import checked_complete, complete, fresh_output, read_json, write_json
from task6.common.randomness import restore_rng, rng_state


def _pending(controller):
    return any(wrapper.router is not None and wrapper.router.pending.sum().item()
               for wrapper in controller.wrappers.values())


def save_checkpoint(path, model, controller, optimizer, scheduler, shuffle_rng, meta):
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    if _pending(controller):
        raise RuntimeError("Cannot checkpoint pending G4 assignment counts")
    staging = path.with_name(f".incomplete_{path.name}_{uuid.uuid4().hex}")
    with fresh_output(staging):
        torch.save(model.state_dict(), staging / "model_state.pt")
        torch.save({
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "shuffle_rng": shuffle_rng.get_state(), "rng": rng_state(),
        }, staging / "training_state.pt")
        write_json(staging / "meta.json", meta)
        complete(staging, meta)
    if path.exists():
        raise FileExistsError(f"Checkpoint was concurrently created: {path}")
    os.rename(staging, path)


def restore_checkpoint(path, model, controller, optimizer=None, scheduler=None, shuffle_rng=None, expected=None):
    import torch

    path = Path(path)
    meta = checked_complete(path)
    if expected is not None:
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Checkpoint {key} mismatch: {path}")
    model_state = torch.load(path / "model_state.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Incomplete model checkpoint: {incompatible}")
    if _pending(controller):
        raise ValueError("Restored checkpoint contains pending G4 counts")
    if optimizer is not None:
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        shuffle_rng.set_state(state["shuffle_rng"])
        restore_rng(state["rng"])
    return meta


def states(config, condition, run_id):
    if not condition.trainable:
        return [{"name": "static", "epoch": 0, "step": 0,
                 "condition": condition.to_dict(), "protocol": protocol_id(config)}]
    directory = run_path(config, "train", condition, run_id) / "checkpoints"
    result = []
    for path in directory.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            meta = read_json(path / "meta.json")
            checked_complete(path, meta)
            if meta["condition"] != condition.to_dict() or meta["protocol"] != protocol_id(config):
                raise ValueError("Checkpoint condition/protocol mismatch")
            if meta["name"] != path.name:
                raise ValueError("Checkpoint directory/name mismatch")
            result.append(meta)
    result.sort(key=lambda item: item["epoch"])
    expected_epochs = list(range(config["training"]["epochs"] + 1))
    if [item["epoch"] for item in result] != expected_epochs:
        raise ValueError(f"Expected all eleven full checkpoints: {directory}")
    expected_names = ["step_0", *[f"step_{epoch}" for epoch in range(1, config["training"]["epochs"])], "final"]
    if [item["name"] for item in result] != expected_names:
        raise ValueError(f"Checkpoint names do not match the scheduled states: {directory}")
    if any(right["step"] <= left["step"] for left, right in zip(result, result[1:])):
        raise ValueError("Checkpoint optimizer steps are not strictly increasing")
    return result


def load_for_capture(config, condition, run_id, state, model, controller, header):
    if condition.trainable:
        path = run_path(config, "train", condition, run_id) / "checkpoints" / state["name"]
        restore_checkpoint(path, model, controller, expected={
            "condition": condition.to_dict(), "protocol": protocol_id(config), "input_header": header,
        })
