from __future__ import annotations

import numpy as np

from task6_phased.common.config import TASKS, protocol_id, routed_root
from task6_phased.common.io import checked_complete, complete, fresh_output, read_json, write_json
from task6_phased.common.provenance import environment
from task6_phased.data.datasets import load_raw, probe_members, tokenize
from task6_phased.substrate.assets import inspect_assets


def artifact_path(config, kind, run_id, experts):
    return routed_root(config) / "artifacts" / kind / f"E_{int(experts)}" / run_id


def header(config, identity):
    return {"schema": 1, "protocol": protocol_id(config), "inputs": identity}


def prepare(config, run_id, experts):
    from task6_phased.dense.model import load_t5
    from task6_phased.routing.routers import raw_centroids
    from task6_phased.substrate.model import ffn_layers
    experts = int(experts)
    paths, labels, artifact_centroids, identity = inspect_assets(config, experts)
    expected = header(config, identity)
    model, tokenizer = load_t5(config, paths["dense_best"], trainable=False)
    probe_dir = artifact_path(config, "probe_sets", run_id, experts)
    with fresh_output(probe_dir):
        for task in TASKS:
            _, validation = load_raw(config, task)
            encoded = tokenize(config, task, validation, tokenizer, "validation", identity["dense"]["model_sha256"])
            count = min(config["tasks"][task]["probe_count"], len(encoded))
            members = probe_members(validation["label"], count, config["capture"]["probe_seed"])
            write_json(probe_dir / f"{task}.json", {
                "task": task, "sample_ids": members.tolist(), "seed": config["capture"]["probe_seed"],
                "algorithm": "PCG64_proportional_stratified_largest_remainder",
                "class_counts": np.bincount(
                    np.asarray(validation["label"])[members],
                    minlength=len(config["tasks"][task]["labels"]),
                ).tolist(),
            })
        write_json(probe_dir / "environment.json", environment())
        complete(probe_dir, expected)
    static = artifact_path(config, "static_routers", run_id, experts)
    with fresh_output(static):
        np.savez(static / "labels.npz", **labels)
        verified = {}
        for key, _, _, module in ffn_layers(model):
            reconstructed = raw_centroids(
                module.wi.weight.detach().cpu(),
                __import__("torch").as_tensor(labels[key], device="cpu"), experts,
            ).numpy()
            # The split writer uses NumPy reduction while this independent
            # reconstruction uses PyTorch. FP32 summation order can differ by
            # a few ULPs across those implementations, so bitwise equality is
            # not a valid cross-runtime identity check. The artifact itself is
            # already content-hashed by its manifest; this check catches a
            # wrong Dense checkpoint or labels without replacing its bytes.
            if not np.allclose(
                reconstructed, artifact_centroids[key], rtol=1.0e-6, atol=1.0e-6,
            ):
                max_abs = float(np.max(np.abs(reconstructed - artifact_centroids[key])))
                raise ValueError(f"{key}: centroid artifact mismatch (max_abs={max_abs})")
            verified[key] = artifact_centroids[key]
        np.savez(static / "centroids.npz", **verified)
        complete(static, expected)
    return expected


def verify_prepared(config, run_id, experts):
    _, _, _, identity = inspect_assets(config, experts)
    expected = header(config, identity)
    checked_complete(artifact_path(config, "probe_sets", run_id, experts), expected)
    checked_complete(artifact_path(config, "static_routers", run_id, experts), expected)
    return expected


def load_context(config, condition, run_id, *, populations=("validation",)):
    from task6_phased.dense.model import load_t5
    from task6_phased.substrate.model import attach
    paths, labels, centroids, identity = inspect_assets(config, condition.experts)
    expected = verify_prepared(config, run_id, condition.experts)
    model, tokenizer = load_t5(config, paths["dense_best"], trainable=False)
    datasets = {}
    for task in TASKS:
        train, validation = load_raw(config, task)
        datasets[task] = {}
        if "train" in populations:
            datasets[task]["train"] = tokenize(config, task, train, tokenizer, "train", identity["dense"]["model_sha256"])
        if "validation" in populations or "probe" in populations:
            encoded = tokenize(config, task, validation, tokenizer, "validation", identity["dense"]["model_sha256"])
            if "validation" in populations:
                datasets[task]["validation"] = encoded
            if "probe" in populations:
                members = read_json(
                    artifact_path(config, "probe_sets", run_id, condition.experts) / f"{task}.json"
                )["sample_ids"]
                datasets[task]["probe"] = encoded.select(members)
    controller = attach(config, condition, model, labels, centroids)
    return model, tokenizer, controller, datasets, expected
