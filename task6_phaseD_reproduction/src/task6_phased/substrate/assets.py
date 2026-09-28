from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from task6_phased.common.config import repository_root, resolved_split, root_for
from task6_phased.common.io import read_json, sha256
from task6_phased.common.provenance import path_identity


def resolve(config, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository_root() / path).resolve()


def model_identity(path):
    path = Path(path)
    from task6_phased.common.config import digest
    return digest({
        file.relative_to(path).as_posix(): sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != "phase_d_dense_manifest.json"
    })


def split_path(config, experts):
    assets = config.get("assets", {})
    mapping = {int(key): value for key, value in assets.get("expert_splits", {}).items()}
    if int(experts) not in mapping:
        raise ValueError(f"Missing assets.expert_splits.{experts}")
    return resolve(config, mapping[int(experts)])


def asset_paths(config, experts):
    assets = config.get("assets", {})
    if "dense_best" not in assets:
        raise ValueError("Missing assets.dense_best")
    result = {
        "dense_best": resolve(config, assets["dense_best"]),
        "expert_split": split_path(config, experts),
    }
    for key, path in result.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {key}: {path}")
        if path.is_relative_to(root_for(config)):
            raise ValueError("Immutable inputs must not live inside output_root")
    return result


@lru_cache(maxsize=8)
def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def inspect_assets(config, experts):
    experts = int(experts)
    paths = asset_paths(config, experts)
    dense_manifest = read_json(paths["dense_best"] / "phase_d_dense_manifest.json")
    if model_identity(paths["dense_best"]) != dense_manifest["model_sha256"]:
        raise ValueError("Exported Dense-MT best hash mismatch")
    split_manifest = read_json(paths["expert_split"] / "manifest.json")
    if split_manifest["dense_model_sha256"] != dense_manifest["model_sha256"]:
        raise ValueError("Split belongs to another Dense-MT checkpoint")
    expected = resolved_split(config, experts)
    expert_size = config["model"]["d_ff"] // experts
    if (
        split_manifest["num_experts"] != experts or split_manifest["expert_size"] != expert_size
        or split_manifest["random_state"] != 1
        or split_manifest["versions"]["k-means-constrained"] != "0.9.1"
        or split_manifest["resolved_parameters"] != expected
    ):
        raise ValueError("Split identity differs from Phase D")
    for name, expected in split_manifest["files"].items():
        if sha256(paths["expert_split"] / name) != expected:
            raise ValueError(f"Split file hash mismatch: {name}")
    labels = _npz(str(paths["expert_split"] / "labels.npz"))
    centroids = _npz(str(paths["expert_split"] / "centroids_raw.npz"))
    if set(labels) != set(centroids) or len(labels) != 12:
        raise ValueError("Split must contain matching labels/centroids for 12 layers")
    for key in labels:
        if labels[key].dtype != np.int64 or labels[key].shape != (2048,):
            raise ValueError(f"{key}: invalid labels")
        if not np.array_equal(
            np.bincount(labels[key], minlength=experts), np.full(experts, expert_size)
        ):
            raise ValueError(f"{key}: invalid equal capacity")
        if centroids[key].shape != (experts, config["model"]["d_model"]):
            raise ValueError(f"{key}: invalid centroid shape")
    identity = {
        "dense": dense_manifest, "split": split_manifest,
        "dense_path": str(paths["dense_best"]), "split_path": str(paths["expert_split"]),
    }
    return paths, labels, centroids, identity
