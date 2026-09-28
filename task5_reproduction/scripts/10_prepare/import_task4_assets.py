"""Audit the original MoEfication inputs; optionally copy them without changing Task 4."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from task5.common.config import load_config
from task5.common.io import read_json, sha256, write_json
from task5.substrate.assets import inspect_task


# Identified against the preposition report, not picked by latest modification time.
ORIGINALS = {
    "sst2": {
        "run": "sst2__seed0__20260819T224020",
        "checkpoint_hash": "e907eb449aea3f795bf5182dcb3a6a5e3fbadb4c9e4f5d22df0101dfe0d39e29",
        "labels_hash": "0da5bf1f31e7f8a7bd1a0a9f4e96c1f63dbe24e3d4914846c52dea4aea443236",
    },
    "mnli": {
        "run": "mnli__seed0__20260819T231255",
        "checkpoint_hash": "53b3704035d0a68c5bb6877138521cad4e5b4ea58b160bafc6c981aeddb1d964",
        "labels_hash": "1b79184ef9cf71ce9e18c1f304f7a6a9f0e33e2c4bf1fc1505e765c826cbacb7",
    },
}


def audit(source):
    import numpy as np
    import pyarrow.parquet as pq

    config = load_config()
    config["assets"] = {}
    records = {}
    copies = []
    for task, original in ORIGINALS.items():
        dense = source / "runs/moefication/train_dense" / task / original["run"] / "checkpoint-best"
        split = (source / "artifacts/moefication/replicate0/expert_splits" / task /
                 original["checkpoint_hash"][:12] / "parameter_seed0")
        data = source / "data/glue" / task
        validation = "validation.parquet" if task == "sst2" else "validation_matched.parquet"
        manifest = read_json(split / "manifest.json")
        if manifest["checkpoint_sha256"] != original["checkpoint_hash"]:
            raise ValueError(f"{task}: checkpoint identity differs from the original report")
        if sha256(split / "labels.npz") != original["labels_hash"]:
            raise ValueError(f"{task}: split labels differ from the original report")
        config["assets"][task] = {"dense": str(dense), "split": str(split), "source": "local_files",
                                  "train": str(data / "train.parquet"), "validation": str(data / validation)}
        _, _, identity = inspect_task(config, task)
        datasets = {}
        for population, filename in (("train", "train.parquet"), ("validation", validation)):
            path = data / filename
            parquet = pq.ParquetFile(path)
            spec = config["tasks"][task]
            if parquet.metadata.num_rows != spec[f"{population}_count"]:
                raise ValueError(f"Unexpected row count: {path}")
            required = {"sentence", "label"} if task == "sst2" else {"premise", "hypothesis", "label"}
            if not required.issubset(parquet.schema_arrow.names):
                raise ValueError(f"Missing dataset fields: {path}")
            labels = parquet.read(columns=["label"]).column("label").to_numpy()
            if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= len(spec["labels"])):
                raise ValueError(f"Invalid label domain: {path}")
            metadata = parquet.schema_arrow.metadata or {}
            features = json.loads(metadata.get(b"huggingface", b"{}"))
            names = features.get("info", {}).get("features", {}).get("label", {}).get("names")
            if names is not None and names != spec["labels"]:
                raise ValueError(f"Label names/order differ: {path}")
            datasets[population] = {"rows": len(labels), "label_counts": np.bincount(labels).tolist(),
                                    "label_names": names, "sha256": sha256(path)}
            copies.append((path, ROOT / "inputs/datasets/glue" / task / filename))
        copies.extend(((dense, ROOT / "inputs/dense" / task), (split, ROOT / "inputs/expert_splits" / task)))
        records[task] = {"source": config["assets"][task], "identity": identity,
                         "checkpoint_sha256": original["checkpoint_hash"], "labels_sha256": original["labels_hash"],
                         "split_replicate": 0, "split_seed": 0, "data": datasets}
    return records, copies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy audited inputs; refuse any existing destination")
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    destination = (ROOT / "inputs").resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Task 4 source and Task 5 inputs must be disjoint")
    records, copies = audit(source)
    print(json.dumps(records, indent=2), flush=True)
    if not args.copy:
        return
    if (destination / "provenance.json").exists() or any(target.exists() for _, target in copies):
        raise FileExistsError("Import refuses to overwrite existing inputs")
    inventory = {}
    for origin, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target)
            pairs = [(p, target / p.relative_to(origin)) for p in sorted(origin.rglob("*")) if p.is_file()]
        else:
            shutil.copy2(origin, target)
            pairs = [(origin, target)]
        for before, after in pairs:
            expected = sha256(before)
            if sha256(after) != expected:
                raise ValueError(f"Copy verification failed: {after}")
            inventory[after.relative_to(destination).as_posix()] = {"sha256": expected, "bytes": after.stat().st_size,
                                                                  "source": str(before)}
    write_json(destination / "provenance.json", {"utc": datetime.now(timezone.utc).isoformat(),
               "tasks": records, "files": inventory, "source_modified": False, "copy_verified": True})
    print(f"Copied and SHA256-verified {len(inventory)} files; source inputs were not modified.")


if __name__ == "__main__":
    main()
