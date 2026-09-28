from __future__ import annotations

from pathlib import Path

import numpy as np

from task8_p01.common.config import TASKS, digest, protocol_id, resolve_path
from task8_p01.common.io import checked_complete, complete, fresh_directory, read_json, write_csv, write_json
from task8_p01.compat.task6 import load_task6_config


def output_root(config):
    return resolve_path(config, config["execution"]["output_root"])


def panel_root(config):
    return output_root(config) / "artifacts/diagnostic_128"


def freeze_panel(config):
    task6_config = load_task6_config(config)
    from task6_phased.data.datasets import load_raw, probe_members

    root = panel_root(config)
    header = {
        "schema": 1,
        "protocol": protocol_id(config),
        "name": "diagnostic_128",
        "count_per_task": 128,
        "seed": 0,
        "algorithm": config["diagnostic"]["selection"],
    }
    if root.exists():
        checked_complete(root, header)
        return read_json(root / "manifest.json")
    manifest = {**header, "tasks": {}}
    all_rows = []
    with fresh_directory(root):
        for task in TASKS:
            _, validation = load_raw(task6_config, task)
            labels = np.asarray(validation["label"], dtype=np.int64)
            members = probe_members(labels, 128, 0)
            rows = [
                {"task": task, "source_index": int(index), "label": int(labels[index])}
                for index in members
            ]
            identity = digest(rows)
            class_counts = np.bincount(labels[members], minlength=len(task6_config["tasks"][task]["labels"])).tolist()
            write_json(root / f"{task}.json", {
                "task": task, "sample_ids": members.tolist(), "hash": identity,
                "class_counts": class_counts,
            })
            manifest["tasks"][task] = {
                "count": len(members), "hash": identity, "class_counts": class_counts,
            }
            all_rows.extend(rows)
        write_csv(root / "panel.csv", all_rows, ("task", "source_index", "label"))
        manifest["panel_hash"] = digest({task: manifest["tasks"][task]["hash"] for task in TASKS})
        write_json(root / "manifest.json", manifest)
        complete(root, header)
    return manifest


def panel_members(config, task):
    root = panel_root(config)
    checked_complete(root)
    value = read_json(root / f"{task}.json")
    if value["task"] != task or len(value["sample_ids"]) != 128:
        raise ValueError(f"Invalid frozen panel for {task}")
    return value["sample_ids"], value["hash"]

