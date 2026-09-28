from __future__ import annotations

import os
from pathlib import Path
import uuid

from task6_phaseb.common.config import protocol_id, run_path
from task6_phaseb.common.io import checked_complete, complete, fresh_output, read_json, sha256, write_json
from task6_phaseb.common.randomness import restore_rng, rng_state


def save_checkpoint(path, controller, optimizer, scheduler, shuffle_rng, meta):
    import torch

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    if any(wrapper.router.pending.sum().item() for wrapper in controller.wrappers.values()):
        raise RuntimeError("Cannot checkpoint pending G4 assignment counts")
    staging = path.with_name(f".incomplete_{path.name}_{uuid.uuid4().hex}")
    with fresh_output(staging):
        torch.save({
            "routers": controller.state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "shuffle_rng": shuffle_rng.get_state(),
            "rng": rng_state(),
        }, staging / "state.pt")
        meta = dict(meta, state_sha256=sha256(staging / "state.pt"))
        write_json(staging / "meta.json", meta)
        complete(staging, meta)
    if path.exists():
        raise FileExistsError(f"Checkpoint was concurrently created: {path}")
    os.rename(staging, path)


def restore_checkpoint(path, controller, optimizer=None, scheduler=None, shuffle_rng=None, expected=None):
    import torch

    meta = checked_complete(path)
    if expected is not None:
        for key, value in expected.items():
            if meta[key] != value:
                raise ValueError(f"Checkpoint {key} mismatch: {path}")
    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    controller.load_state(state["routers"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        shuffle_rng.set_state(state["shuffle_rng"])
        restore_rng(state["rng"])
    return meta


def states(config, condition, run_id):
    if not condition.trainable:
        return [{
            "name": "static",
            "epoch": 0,
            "step": 0,
            "condition": condition.to_dict(),
            "protocol": protocol_id(config),
        }]
    directory = run_path(config, "train", condition, run_id) / "checkpoints"
    result = []
    for path in directory.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            meta = read_json(path / "meta.json")
            checked_complete(path, meta)
            if meta["condition"] != condition.to_dict() or meta["protocol"] != protocol_id(config):
                raise ValueError("Checkpoint condition/protocol mismatch")
            result.append(meta)
    result.sort(key=lambda state: state["epoch"])
    expected_epochs = list(range(config["training"]["epochs"] + 1))
    if [state["epoch"] for state in result] != expected_epochs:
        raise ValueError(f"Expected every checkpoint epoch 0..{config['training']['epochs']}: {directory}")
    if any(right["step"] <= left["step"] for left, right in zip(result, result[1:])):
        raise ValueError("Checkpoint steps are not strictly increasing")
    return result


def load_for_capture(config, condition, run_id, state, controller, header):
    if condition.trainable:
        path = run_path(config, "train", condition, run_id) / "checkpoints" / state["name"]
        restore_checkpoint(path, controller, expected={
            "condition": condition.to_dict(),
            "protocol": protocol_id(config),
            "input_header": header,
        })
