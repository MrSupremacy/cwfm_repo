from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np

from task6.common.config import digest, repository_root, root_for
from task6.common.io import hash_files, read_json, sha256


def asset_paths(config, task):
    if task not in config.get("assets", {}):
        raise ValueError(f"Missing assets.{task}; supply --local configs/local/server.yaml")
    a = dict(config["assets"][task])
    for key in ("dense", "split", "train", "validation", "dataset"):
        if key in a:
            a[key] = (repository_root() / a[key]).resolve()
            if not a[key].exists():
                raise FileNotFoundError(f"Missing remote input {task}.{key}: {a[key]}")
            source = a[key] if a[key].is_dir() else a[key].parent
            output = root_for(config)
            if output.is_relative_to(source) or any(a[key].is_relative_to(output / name) for name in ("tmp", "runs", "artifacts", "results")):
                raise ValueError("Read-only inputs and generated output/cache directories must not overlap")
    return a


@lru_cache(maxsize=32)
def cached_file_hash(path, size, modified):
    return sha256(path)


def file_hash(path):
    p = Path(path)
    stat = p.stat()
    return cached_file_hash(str(p), stat.st_size, stat.st_mtime_ns)


def path_identity(path):
    path = Path(path)
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return digest({p.relative_to(path).as_posix() if path.is_dir() else p.name: file_hash(p) for p in paths})


def validate_labels(labels, d_ff, experts, size):
    a = np.asarray(labels)
    if a.shape != (d_ff,) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Split labels must be an integer vector in original neuron order")
    if np.any(a < 0) or np.any(a >= experts):
        raise ValueError("Expert label out of range")
    if not np.array_equal(np.bincount(a, minlength=experts), np.full(experts, size)):
        raise ValueError("Split must cover every neuron with exactly equal expert sizes")


def inspect_task(config, task):
    a = asset_paths(config, task)
    dense, split = a["dense"], a["split"]
    manifest = read_json(split / "manifest.json")
    # Compatibility fingerprint for the existing MoEfication artifact, not a dependency on its code.
    files = [dense / n for n in ("config.json", "model.safetensors", "pytorch_model.bin") if (dense / n).is_file()]
    if not (dense / "config.json").is_file() or len(files) < 2:
        raise ValueError("Expected the existing unsharded T5-small checkpoint and config.json")
    if hash_files(dense, files) != manifest["checkpoint_sha256"]:
        raise ValueError("Expert split does not belong to this dense checkpoint")
    if manifest["task"] != task or manifest["method"] != "parameter":
        raise ValueError("Expected the selected task's balanced parameter K-Means artifact")
    if file_hash(split / "labels.npz") != manifest["files"]["labels.npz"]:
        raise ValueError("Split labels hash mismatch")
    m = config["model"]
    expected = {f"{stack}_layer_{i:02d}" for stack in ("encoder", "decoder") for i in range(m[f"{stack}_layers"])}
    with np.load(split / "labels.npz", allow_pickle=False) as archive:
        labels = {key: archive[key].copy() for key in archive.files}
    if set(labels) != expected:
        raise ValueError("Expected exactly the configured encoder/decoder layer keys")
    for values in labels.values():
        validate_labels(values, m["d_ff"], m["num_experts"], m["expert_size"])
    input_data = {k: path_identity(a[k]) for k in ("train", "validation", "dataset") if k in a}
    identity = {"dense": path_identity(dense), "split": path_identity(split), "data": input_data}
    return a, labels, identity
