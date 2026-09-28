from __future__ import annotations

import importlib.metadata

import numpy as np

from task6_phaseb.common.config import protocol_id, root_for, validate_run_id
from task6_phaseb.common.io import checked_complete, complete, fresh_output, read_json, write_json
from task6_phaseb.data.datasets import expected_probe_keys, load_raw, probe_members, tokenize
from task6_phaseb.substrate.assets import inspect_task


def shared_path(config, kind, task, experts, run_id):
    validate_run_id(run_id)
    return root_for(config) / "artifacts" / kind / task / f"E_{experts}" / run_id


def input_header(config, task, experts, identity):
    return {
        "protocol": protocol_id(config),
        "task": task,
        "experts": experts,
        "inputs": identity,
        "schema": 1,
    }


def environment():
    import platform
    import torch

    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "torch", "transformers", "datasets", "pyarrow")
    }
    return {
        "python": platform.python_version(),
        "versions": versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def verify_prepared(config, task, experts, run_id):
    _, _, _, identity = inspect_task(config, task, experts)
    header = input_header(config, task, experts, identity)
    checked_complete(shared_path(config, "probe_sets", task, experts, run_id), header)
    checked_complete(shared_path(config, "static_routers", task, experts, run_id), header)
    return header


def prepare_task(config, task, experts, run_id):
    import torch

    from task6_phaseb.routing.routers import raw_centroids
    from task6_phaseb.substrate.model import ffn_layers, load_dense

    paths, labels, artifact_centroids, identity = inspect_task(config, task, experts)
    header = input_header(config, task, experts, identity)
    model, tokenizer = load_dense(config, paths["dense"])
    train, validation = load_raw(config, task, paths)
    token_identity = {"dense": identity["dense"], "data": identity["data"]}
    encoded = tokenize(config, task, validation, tokenizer, token_identity, "validation")
    count = min(config["tasks"][task]["probe_count"], len(encoded))
    members = probe_members(validation["label"], count, config["capture"]["probe_seed"])
    probe = encoded.select(members.tolist())

    probe_dir = shared_path(config, "probe_sets", task, experts, run_id)
    with fresh_output(probe_dir):
        write_json(probe_dir / "members.json", {
            "sample_ids": members.tolist(),
            "algorithm": "PCG64_largest_remainder_v1",
            "numpy_version": np.__version__,
            "seed": config["capture"]["probe_seed"],
            "class_counts": np.bincount(
                np.asarray(validation["label"])[members],
                minlength=len(config["tasks"][task]["labels"]),
            ).tolist(),
            "expected_keys": expected_probe_keys(
                probe, config["capture"]["probe_batch_size"], tokenizer.padding_side
            ),
        })
        write_json(probe_dir / "context.json", {
            "header": header,
            "config": config,
            "environment": environment(),
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "special_tokens": tokenizer.special_tokens_map,
            },
            "generation_config": model.generation_config.to_dict(),
        })
        complete(probe_dir, header)

    tokenize(config, task, train, tokenizer, token_identity, "train")
    static_dir = shared_path(config, "static_routers", task, experts, run_id)
    with fresh_output(static_dir):
        np.savez(static_dir / "labels.npz", **labels)
        centroids = {}
        for key, _, _, module in ffn_layers(model):
            # The immutable split artifact was generated and audited with a CPU
            # float32 reduction. Repeating the mean on CUDA can differ in the
            # final bits solely because the reduction order is device-specific,
            # which makes an exact artifact check fail despite identical inputs.
            # Recompute on CPU so the strict byte-level numerical contract stays
            # meaningful; attach() moves the verified snapshot back to the model
            # device when it constructs each router.
            value = raw_centroids(
                module.wi.weight.detach().cpu(),
                torch.as_tensor(labels[key], device="cpu"),
                experts,
            ).numpy()
            if not np.array_equal(value, artifact_centroids[key]):
                max_abs = float(np.max(np.abs(value - artifact_centroids[key])))
                raise ValueError(
                    f"Raw centroid does not match split artifact: {key}; max_abs={max_abs}"
                )
            centroids[key] = artifact_centroids[key].copy()
        np.savez(static_dir / "centroids.npz", **centroids)
        write_json(static_dir / "source_manifest.json", read_json(paths["split"] / "manifest.json"))
        complete(static_dir, header)


def load_context(config, condition, run_id, *, training=False):
    from task6_phaseb.substrate.model import attach, load_dense

    task, experts = condition.task, condition.experts
    paths, labels, _, identity = inspect_task(config, task, experts)
    probe_dir = shared_path(config, "probe_sets", task, experts, run_id)
    static_dir = shared_path(config, "static_routers", task, experts, run_id)
    header = input_header(config, task, experts, identity)
    checked_complete(probe_dir, header)
    checked_complete(static_dir, header)
    recorded = read_json(probe_dir / "context.json")["environment"]
    current = environment()
    if recorded["versions"] != current["versions"] or recorded["python"] != current["python"]:
        raise ValueError("Model execution environment differs from preparation")
    model, tokenizer = load_dense(config, paths["dense"])
    train, validation = load_raw(config, task, paths)
    token_identity = {"dense": identity["dense"], "data": identity["data"]}
    dataset = tokenize(
        config,
        task,
        train if training else validation,
        tokenizer,
        token_identity,
        "train" if training else "validation",
    )
    with np.load(static_dir / "centroids.npz", allow_pickle=False) as archive:
        centroids = {key: archive[key].copy() for key in archive.files}
    controller = attach(config, condition, model, labels, centroids)
    members = read_json(probe_dir / "members.json")
    return model, tokenizer, controller, dataset, header, members
