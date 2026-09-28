from __future__ import annotations

import importlib.metadata

import numpy as np

from task6.common.config import protocol_id, root_for, validate_run_id
from task6.common.io import checked_complete, complete, fresh_output, read_json, write_json
from task6.data.datasets import expected_probe_keys, load_raw, probe_members, tokenize
from task6.substrate.assets import inspect_task


def shared_path(config, kind, task, run_id):
    validate_run_id(run_id)
    return root_for(config) / "artifacts" / kind / task / run_id


def input_header(config, task, identity):
    return {"protocol": protocol_id(config), "task": task, "inputs": identity, "schema": 1}


def environment():
    import platform
    import torch

    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "transformers", "datasets", "pyarrow")}
    return {
        "python": platform.python_version(), "versions": versions, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def verify_prepared(config, task, run_id):
    _, _, identity = inspect_task(config, task)
    expected = input_header(config, task, identity)
    probe = shared_path(config, "probe_sets", task, run_id)
    static = shared_path(config, "static_routers", task, run_id)
    checked_complete(probe, expected)
    checked_complete(static, expected)
    return expected


def prepare_task(config, task, run_id):
    import torch

    from task6.routing.routers import raw_centroids
    from task6.substrate.model import ffn_layers, load_dense

    paths, labels, identity = inspect_task(config, task)
    header = input_header(config, task, identity)
    model, tokenizer = load_dense(config, paths["dense"])
    train, validation = load_raw(config, task, paths)
    encoded = tokenize(config, task, validation, tokenizer, identity, "validation")
    count = min(config["tasks"][task]["probe_count"], len(encoded))
    members = probe_members(validation["label"], count, config["capture"]["probe_seed"])
    probe = encoded.select(members.tolist())

    directory = shared_path(config, "probe_sets", task, run_id)
    with fresh_output(directory):
        write_json(directory / "members.json", {
            "sample_ids": members.tolist(), "algorithm": "PCG64_largest_remainder_v1",
            "numpy_version": np.__version__, "seed": config["capture"]["probe_seed"],
            "class_counts": np.bincount(np.asarray(validation["label"])[members],
                                        minlength=len(config["tasks"][task]["labels"])).tolist(),
            "expected_keys": expected_probe_keys(probe, config["capture"]["probe_batch_size"], tokenizer.padding_side),
        })
        write_json(directory / "context.json", {
            "header": header, "config": config, "environment": environment(),
            "tokenizer": {"class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side,
                          "truncation_side": tokenizer.truncation_side,
                          "special_tokens": tokenizer.special_tokens_map},
            "generation_config": model.generation_config.to_dict(),
        })
        complete(directory, header)

    tokenize(config, task, train, tokenizer, identity, "train")
    directory = shared_path(config, "static_routers", task, run_id)
    with fresh_output(directory):
        np.savez(directory / "labels.npz", **labels)
        centroids = {
            key: raw_centroids(module.wi.weight, torch.as_tensor(labels[key], device=module.wi.weight.device),
                               config["model"]["num_experts"]).cpu().numpy()
            for key, _, _, module in ffn_layers(model)
        }
        np.savez(directory / "centroids.npz", **centroids)
        complete(directory, header)


def load_context(config, condition, run_id, *, training=False):
    import torch

    from task6.substrate.model import attach, load_dense

    task = condition.task
    paths, labels, identity = inspect_task(config, task)
    header = verify_prepared(config, task, run_id)
    recorded = read_json(shared_path(config, "probe_sets", task, run_id) / "context.json")["environment"]
    current = environment()
    if recorded["versions"] != current["versions"] or recorded["python"] != current["python"]:
        raise ValueError("Model execution environment differs from preparation")
    model, tokenizer = load_dense(config, paths["dense"])
    train, validation = load_raw(config, task, paths)
    population = "train" if training else "validation"
    dataset = tokenize(config, task, train if training else validation, tokenizer, identity, population)
    static = shared_path(config, "static_routers", task, run_id)
    with np.load(static / "centroids.npz", allow_pickle=False) as archive:
        centroids = {key: archive[key].copy() for key in archive.files}
    controller = attach(config, condition, model, labels, centroids)
    members = read_json(shared_path(config, "probe_sets", task, run_id) / "members.json")
    return model, tokenizer, controller, dataset, header, members
