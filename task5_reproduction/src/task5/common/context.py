from __future__ import annotations

import importlib.metadata
import numpy as np

from task5.common.config import digest, extension_spec, protocol_id, root_for, validate_run_id
from task5.common.io import checked_complete, complete, fresh_output, read_json, write_json
from task5.data.datasets import expected_probe_keys, load_raw, probe_members, tokenize
from task5.substrate.assets import inspect_task


def shared_path(config, kind, task, run_id):
    validate_run_id(run_id)
    return root_for(config) / "artifacts" / kind / task / run_id


def input_header(config, task, identity):
    return {"protocol": protocol_id(config), "task": task, "inputs": identity, "schema": 1}


def _shared_config_identity(config):
    """Identity of settings that can affect prepared probe/static artifacts."""
    ignored = {"assets", "execution", "extension", "metrics", "variants"}
    return digest({key: value for key, value in config.items() if key not in ignored})


def _prepared_header(config, task, run_id, identity, probe_dir, static_dir):
    current = input_header(config, task, identity)
    extension = extension_spec(config, run_id)
    if extension is None:
        checked_complete(probe_dir, current)
        checked_complete(static_dir, current)
        return current

    probe_header = checked_complete(probe_dir)
    static_header = checked_complete(static_dir)
    if probe_header != static_header:
        raise ValueError("Prepared probe and static-router identities differ")
    expected_base = {**current, "protocol": extension["base_protocol"]}
    if probe_header != expected_base:
        raise ValueError("Existing prepared artifacts do not match the declared base protocol and current inputs")
    recorded_config = read_json(probe_dir / "context.json")["config"]
    if _shared_config_identity(recorded_config) != _shared_config_identity(config):
        raise ValueError("Shared preparation settings differ from the declared base run")
    return probe_header


def verify_prepared(config, task, run_id):
    """Hash-check prepared inputs without loading a model or dataset into memory."""
    _, _, identity = inspect_task(config, task)
    return _prepared_header(config, task, run_id, identity,
                            shared_path(config, "probe_sets", task, run_id),
                            shared_path(config, "static_routers", task, run_id))


def environment():
    import platform
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "datasets", "pyarrow")}
    import torch
    return {"python": platform.python_version(), "versions": versions, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def prepare_task(config, task, run_id):
    from task5.substrate.model import ffn_layers, load_dense
    from task5.routing.routers import make_hash_table, raw_centroids
    import torch
    paths, labels, identity = inspect_task(config, task)
    header = input_header(config, task, identity)
    model, tokenizer = load_dense(config, paths["dense"])
    train, validation = load_raw(config, task, paths)
    encoded = tokenize(config, task, validation, tokenizer, identity, "validation")
    n = min(config["tasks"][task]["probe_count"], len(encoded))
    members = probe_members(validation["label"], n, config["capture"]["probe_seed"])
    probe = encoded.select(members.tolist())
    directory = shared_path(config, "probe_sets", task, run_id)
    with fresh_output(directory):
        write_json(directory / "members.json", {"sample_ids": members.tolist(), "algorithm": "PCG64_largest_remainder_v1",
                    "numpy_version": np.__version__, "seed": config["capture"]["probe_seed"],
                    "class_counts": np.bincount(np.asarray(validation["label"])[members], minlength=len(config["tasks"][task]["labels"])).tolist(),
                    "expected_keys": expected_probe_keys(probe, config["capture"]["probe_batch_size"], tokenizer.padding_side)})
        write_json(directory / "context.json", {"header": header, "config": config, "environment": environment(),
                    "tokenizer": {"class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side,
                                  "truncation_side": tokenizer.truncation_side, "special_tokens": tokenizer.special_tokens_map},
                    "generation_config": model.generation_config.to_dict()})
        complete(directory, header)
    # Prepare the training cache once before launching independent GPU runs.
    # All later map calls use the same verified explicit cache, never the input directory.
    tokenize(config, task, train, tokenizer, identity, "train")
    directory = shared_path(config, "static_routers", task, run_id)
    with fresh_output(directory):
        # Tiny immutable mapping snapshot: offline metrics need no remote model/dataset.
        np.savez(directory / "labels.npz", **labels)
        centroids = {key: raw_centroids(module.wi.weight, torch.as_tensor(labels[key], device=module.wi.weight.device),
                                     config["model"]["num_experts"]).cpu().numpy() for key, _, _, module in ffn_layers(model)}
        np.savez(directory / "centroids.npz", **centroids)
        for seed in config["suite"]["seeds"]:
            table = make_hash_table(model.config.vocab_size, config["model"]["num_experts"], seed)
            np.savez(directory / f"hash_seed_{seed}.npz", table=table.numpy())
        complete(directory, header)


def load_context(config, condition, run_id, *, training=False):
    from task5.substrate.model import attach, load_dense
    import torch
    task = condition.task
    paths, labels, identity = inspect_task(config, task)
    probe_dir = shared_path(config, "probe_sets", task, run_id)
    static_dir = shared_path(config, "static_routers", task, run_id)
    header = _prepared_header(config, task, run_id, identity, probe_dir, static_dir)
    recorded = read_json(probe_dir / "context.json")["environment"]
    current = environment()
    if recorded["versions"] != current["versions"] or recorded["python"] != current["python"]:
        raise ValueError("Model execution environment differs from preparation; align versions before continuing")
    model, tokenizer = load_dense(config, paths["dense"])
    train, validation = load_raw(config, task, paths)
    dataset = tokenize(config, task, train if training else validation, tokenizer, identity,
                       "train" if training else "validation")
    with np.load(static_dir / "centroids.npz", allow_pickle=False) as archive:
        centroids = {key: archive[key].copy() for key in archive.files}
    table = None
    if condition.arm == "G0":
        with np.load(static_dir / f"hash_seed_{condition.seed}.npz", allow_pickle=False) as archive:
            table = torch.from_numpy(archive["table"].copy())
    controller = attach(config, condition, model, labels, centroids, table)
    members = read_json(probe_dir / "members.json")
    return model, tokenizer, controller, dataset, header, members
