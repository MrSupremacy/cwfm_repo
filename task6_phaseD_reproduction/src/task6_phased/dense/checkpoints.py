from __future__ import annotations

import os
from pathlib import Path
import uuid

from task6_phased.common.io import complete, fresh_output, read_json, checked_complete, write_json
from task6_phased.common.randomness import rng_state, restore_rng


def save_dense_checkpoint(path, model, tokenizer, optimizer, scheduler, stream, meta):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    staging = path.with_name(f".incomplete_{path.name}_{uuid.uuid4().hex}")
    with fresh_output(staging):
        model.save_pretrained(staging / "model", safe_serialization=True)
        tokenizer.save_pretrained(staging / "model")
        torch.save({
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "sampler": stream.state_dict(), "rng": rng_state(),
        }, staging / "trainer.pt")
        write_json(staging / "meta.json", meta)
        complete(staging, meta)
    os.rename(staging, path)


def load_dense_checkpoint(path, model, optimizer, scheduler, stream, expected):
    import torch
    path = Path(path)
    meta = checked_complete(path)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Dense checkpoint {key} mismatch")
    from safetensors.torch import load_file
    state = load_file(path / "model/model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True)
    trainer = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    stream.load_state_dict(trainer["sampler"])
    restore_rng(trainer["rng"])
    return meta


def dense_states(directory, total_steps, every, expected=None):
    result = []
    for step in range(0, total_steps + 1, every):
        name = f"step_{step}"
        path = Path(directory) / name
        header = checked_complete(path)
        if int(header["step"]) != step:
            raise ValueError(f"Checkpoint step mismatch: {path}")
        for key, value in (expected or {}).items():
            if header.get(key) != value:
                raise ValueError(f"Checkpoint {key} mismatch: {path}")
        result.append({"name": name, "step": step, "path": path})
    return result
