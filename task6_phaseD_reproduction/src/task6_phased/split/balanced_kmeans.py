from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from task6_phased.common.io import fresh_output, sha256, write_json
from task6_phased.common.config import resolved_split
from task6_phased.common.provenance import path_identity
from task6_phased.dense.model import load_t5, resolve_asset
from task6_phased.substrate.assets import split_path
from task6_phased.substrate.model import ffn_layers


def _require_versions(config):
    expected = {
        "k-means-constrained": config["split_generation"]["k_means_constrained_version"],
        "scikit-learn": config["split_generation"]["scikit_learn_version"],
        "numpy": config["split_generation"]["numpy_version"],
        "scipy": config["split_generation"]["scipy_version"],
        "ortools": config["split_generation"]["ortools_version"],
        "joblib": config["split_generation"]["joblib_version"],
    }
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Split environment mismatch: expected={expected}, actual={actual}")
    return actual


def generate_split(config, experts):
    from k_means_constrained import KMeansConstrained
    from sklearn.preprocessing import normalize
    versions = _require_versions(config)
    dense = resolve_asset(config, "dense_best")
    manifest_path = dense / "phase_d_dense_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Dense best was not exported through the Phase D gate")
    from task6_phased.common.io import read_json
    dense_manifest = read_json(manifest_path)
    model, _ = load_t5(config, dense, trainable=False)
    experts = int(experts)
    target = split_path(config, experts)
    split = resolved_split(config, experts)
    with fresh_output(target):
        labels, raw_centroids, cosine_centroids, diagnostics = {}, {}, {}, {}
        for key, _, _, module in ffn_layers(model):
            raw = module.wi.weight.detach().float().cpu().numpy()
            normalized = normalize(raw, norm="l2", axis=1)
            estimator = KMeansConstrained(
                n_clusters=split["n_clusters"], size_min=split["size_min"],
                size_max=split["size_max"], random_state=split["random_state"],
                init=split["init"], n_init=split["n_init"], max_iter=split["max_iter"],
                tol=split["tol"], copy_x=split["copy_x"], n_jobs=split["n_jobs"],
            )
            assignment = estimator.fit_predict(normalized).astype(np.int64)
            counts = np.bincount(assignment, minlength=split["n_clusters"])
            if not np.array_equal(counts, np.full(split["n_clusters"], split["size_min"])):
                raise ValueError(
                    f"{key}: constrained clustering did not produce "
                    f"{experts}x{split['size_min']}"
                )
            labels[key] = assignment
            centroid = np.stack([raw[assignment == expert].mean(0) for expert in range(split["n_clusters"])]).astype(np.float32)
            raw_centroids[key] = centroid
            norms = np.linalg.norm(centroid, axis=1, keepdims=True)
            cosine_centroids[key] = (centroid / np.maximum(norms, 1.0e-12)).astype(np.float32)
            diagnostics[key] = {
                "inertia": float(estimator.inertia_), "iterations": int(estimator.n_iter_),
                "cluster_sizes": counts.tolist(),
            }
        np.savez(target / "labels.npz", **labels)
        np.savez(target / "centroids_raw.npz", **raw_centroids)
        np.savez(target / "centroids_cosine.npz", **cosine_centroids)
        write_json(target / "diagnostics.json", diagnostics)
        files = {
            name: sha256(target / name)
            for name in ("labels.npz", "centroids_raw.npz", "centroids_cosine.npz", "diagnostics.json")
        }
        write_json(target / "manifest.json", {
            "format": "task6_phased_split_v1", "method": "parameter",
            "num_experts": experts, "expert_size": split["size_min"],
            "d_ff": config["model"]["d_ff"],
            "random_state": 1, "resolved_parameters": split, "versions": versions,
            "dense_manifest": dense_manifest, "dense_model_sha256": dense_manifest["model_sha256"],
            "files": files,
        })
    return target
