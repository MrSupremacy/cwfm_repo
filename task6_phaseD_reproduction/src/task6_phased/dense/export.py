from __future__ import annotations

import shutil

from task6_phased.common.config import dense_run_path, protocol_id
from task6_phased.common.io import checked_complete, read_json, write_json
from task6_phased.common.provenance import path_identity
from task6_phased.dense.model import resolve_asset


def export_dense_best(config, run_id):
    selection = read_json(dense_run_path(config, "evaluation", run_id) / "selection.json")
    if selection["protocol"] != protocol_id(config):
        raise ValueError("Dense selection belongs to another protocol")
    checkpoint = dense_run_path(config, "train", run_id) / "checkpoints" / selection["state"]["name"]
    checkpoint_header = checked_complete(checkpoint)
    expected = {
        "protocol": protocol_id(config), "run_id": run_id,
        "total_steps": config["dense"]["total_optimizer_steps"],
        "name": selection["state"]["name"], "step": selection["state"]["step"],
        "domain_exposure": {
            task: selection["state"]["step"] * config["dense"]["domain_batch_size"]
            for task in ("sst2", "mnli", "qnli", "qqp")
        },
    }
    for key, value in expected.items():
        if checkpoint_header.get(key) != value:
            raise ValueError(f"Selected Dense checkpoint {key} mismatch")
    source = checkpoint / "model"
    target = resolve_asset(config, "dense_best")
    if target.exists():
        raise FileExistsError(f"Refusing to replace exported Dense best: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    manifest = {
        "format": "task6_phased_dense_best_v1", "source_run_id": run_id,
        "source_state": selection["state"], "protocol": protocol_id(config),
        "model_sha256": path_identity(target),
    }
    write_json(target / "phase_d_dense_manifest.json", manifest)
    return manifest
