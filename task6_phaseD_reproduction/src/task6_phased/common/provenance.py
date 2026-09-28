from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

from task6_phased.common.config import digest
from task6_phased.common.io import sha256


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "pyarrow", "scikit-learn", "k-means-constrained"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        cuda = torch.version.cuda
        device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    except ImportError:
        cuda, device = None, "unavailable"
    return {"python": platform.python_version(), "packages": packages, "cuda": cuda, "device": device}


def path_identity(path):
    path = Path(path)
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    return digest({
        item.relative_to(path).as_posix() if path.is_dir() else item.name: sha256(item)
        for item in files
    })
