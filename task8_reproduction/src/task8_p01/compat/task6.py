from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from task8_p01.common.config import resolve_path


def task6_root(config):
    root = resolve_path(config, config["execution"]["task6_repo"])
    if not (root / "src/task6_phased/__init__.py").is_file():
        raise FileNotFoundError(f"Not a Task 6 Phase D source tree: {root}")
    return root


def bootstrap_task6(config):
    root = task6_root(config)
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return root


def load_task6_config(config):
    root = bootstrap_task6(config)
    from task6_phased.common.config import load_config

    suite = Path(config["execution"]["task6_suite"])
    local = Path(config["execution"]["task6_local"])
    suite = suite if suite.is_absolute() else root / suite
    local = local if local.is_absolute() else root / local
    return load_config(suite, local)


def task6_source_identity(config):
    root = task6_root(config)
    files = sorted((root / "src/task6_phased").rglob("*.py"))
    per_file = {}
    overall = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes().replace(b"\r\n", b"\n")
        value = hashlib.sha256(content).hexdigest()
        per_file[relative] = value
        overall.update(relative.encode())
        overall.update(bytes.fromhex(value))
    return {"root": str(root), "sha256": overall.hexdigest(), "files": per_file}

