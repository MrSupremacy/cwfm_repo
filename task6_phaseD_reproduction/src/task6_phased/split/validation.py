from __future__ import annotations

import numpy as np

from task6_phased.common.config import resolved_split
from task6_phased.common.io import read_json, sha256
from task6_phased.substrate.assets import split_path


def validate_split(config, experts):
    experts = int(experts)
    path = split_path(config, experts)
    manifest = read_json(path / "manifest.json")
    expected = resolved_split(config, experts)
    expert_size = config["model"]["d_ff"] // experts
    if (
        manifest["num_experts"] != experts or manifest["expert_size"] != expert_size
        or manifest["random_state"] != 1
        or manifest["versions"]["k-means-constrained"] != "0.9.1"
        or manifest["resolved_parameters"] != expected
    ):
        raise ValueError("Split manifest differs from Phase D protocol")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Split artifact hash mismatch: {name}")
    with np.load(path / "labels.npz", allow_pickle=False) as archive:
        if len(archive.files) != 12:
            raise ValueError("Expected 12 FFN layer splits")
        for key in archive.files:
            values = archive[key]
            if values.dtype != np.int64 or values.shape != (2048,):
                raise ValueError(f"{key}: labels schema mismatch")
            if not np.array_equal(
                np.bincount(values, minlength=experts), np.full(experts, expert_size)
            ):
                raise ValueError(f"{key}: labels are not exact {experts}x{expert_size}")
    return manifest
