from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from task6_phaseb.common.config import digest, repository_root, root_for
from task6_phaseb.common.io import hash_files, read_json, sha256


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository_root() / path).resolve()


def asset_paths(config, task, experts):
    if task not in config.get("assets", {}):
        raise ValueError(f"Missing assets.{task}; supply a local/server YAML")
    raw = config["assets"][task]
    result = {"source": raw["source"]}
    for key in ("dense", "train", "validation", "dataset"):
        if key in raw:
            result[key] = _resolve(raw[key])
    split_mapping = {int(key): value for key, value in raw.get("splits", {}).items()}
    if experts not in split_mapping:
        raise ValueError(f"Missing assets.{task}.splits.{experts}")
    result["split"] = _resolve(split_mapping[experts])

    output = root_for(config)
    for key, path in result.items():
        if key == "source":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing input {task}/E{experts}.{key}: {path}")
        source = path if path.is_dir() else path.parent
        if output.is_relative_to(source) or any(
            path.is_relative_to(output / name) for name in ("tmp", "runs", "artifacts", "results")
        ):
            raise ValueError("Read-only inputs and generated outputs must not overlap")
    return result


@lru_cache(maxsize=128)
def cached_file_hash(path, size, modified):
    del size, modified
    return sha256(path)


def file_hash(path):
    path = Path(path)
    stat = path.stat()
    return cached_file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def path_identity(path):
    path = Path(path)
    paths = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    return digest({
        item.relative_to(path).as_posix() if path.is_dir() else item.name: file_hash(item)
        for item in paths
    })


def validate_labels(labels, d_ff, experts, size):
    values = np.asarray(labels)
    if values.shape != (d_ff,) or values.dtype != np.int64:
        raise ValueError("Split labels must be int64 in original neuron order")
    if np.any(values < 0) or np.any(values >= experts):
        raise ValueError("Expert label out of range")
    if not np.array_equal(np.bincount(values, minlength=experts), np.full(experts, size)):
        raise ValueError("Split must cover every neuron with exact equal capacity")


def inspect_task(config, task, experts):
    paths = asset_paths(config, task, experts)
    dense, split = paths["dense"], paths["split"]
    manifest = read_json(split / "manifest.json")
    model_files = [
        dense / name for name in ("config.json", "model.safetensors", "pytorch_model.bin")
        if (dense / name).is_file()
    ]
    if not (dense / "config.json").is_file() or len(model_files) < 2:
        raise ValueError("Expected an unsharded T5-small checkpoint and config.json")
    if hash_files(dense, model_files) != manifest["checkpoint_sha256"]:
        raise ValueError("Expert split does not belong to this dense checkpoint")
    expected_size = config["model"]["d_ff"] // experts
    if (
        manifest["task"] != task
        or manifest["method"] != "parameter"
        or manifest["num_experts"] != experts
        or manifest["expert_size"] != expected_size
        or manifest["d_ff"] != config["model"]["d_ff"]
    ):
        raise ValueError("Split manifest disagrees with task/E/equal-capacity protocol")
    for name in ("labels.npz", "centroids_raw.npz", "centroids_cosine.npz", "diagnostics.json"):
        if file_hash(split / name) != manifest["files"][name]:
            raise ValueError(f"Split file hash mismatch: {name}")

    layer_names = {
        f"{stack}_layer_{index:02d}"
        for stack in ("encoder", "decoder")
        for index in range(config["model"][f"{stack}_layers"])
    }
    with np.load(split / "labels.npz", allow_pickle=False) as archive:
        labels = {key: archive[key].copy() for key in archive.files}
    with np.load(split / "centroids_raw.npz", allow_pickle=False) as archive:
        centroids = {key: archive[key].copy() for key in archive.files}
    if set(labels) != layer_names or set(centroids) != layer_names:
        raise ValueError("Split layer keys do not match configured encoder/decoder layers")
    for key in layer_names:
        validate_labels(labels[key], config["model"]["d_ff"], experts, expected_size)
        if centroids[key].shape != (experts, config["model"]["d_model"]) or centroids[key].dtype != np.float32:
            raise ValueError(f"Invalid raw centroid array: {key}")
        if not np.isfinite(centroids[key]).all():
            raise ValueError(f"Non-finite raw centroid: {key}")

    data_identity = {
        key: path_identity(paths[key]) for key in ("train", "validation", "dataset") if key in paths
    }
    identity = {
        "task": task,
        "experts": experts,
        "dense": path_identity(dense),
        "split": path_identity(split),
        "split_labels": manifest["files"]["labels.npz"],
        "checkpoint": manifest["checkpoint_sha256"],
        "data": data_identity,
    }
    return paths, labels, centroids, identity
