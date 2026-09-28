from __future__ import annotations

import os
from pathlib import Path
import shutil
from urllib.request import urlopen
import uuid

from task6_phased.common.config import TASKS, input_root
from task6_phased.data.datasets import parquet_sha256


BASE = "https://huggingface.co/datasets/nyu-mll/glue/resolve/refs%2Fconvert%2Fparquet"


def _fetch(url, target):
    with urlopen(url) as source, Path(target).open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def download_glue(config):
    """Download exact pinned Parquet artifacts; never leave a partial task directory."""
    root = input_root(config) / "data/glue"
    root.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        target = root / task
        validation_split = config["tasks"][task]["validation_split"]
        validation_file = "validation_matched.parquet" if task == "mnli" else "validation.parquet"
        expected_files = {"train": "train.parquet", "validation": validation_file}
        if target.exists():
            if not all((target / name).is_file() for name in expected_files.values()):
                raise FileExistsError(f"Existing dataset directory is incomplete; inspect manually: {target}")
            for population, name in expected_files.items():
                expected = config["data"]["parquet_sha256"][task][population]
                if parquet_sha256(target / name) != expected:
                    raise ValueError(f"Existing {task}.{population} parquet hash mismatch")
            print(f"Reuse verified {task} parquet directory: {target}")
            continue
        staging = root / f".incomplete_{task}_{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            sources = {
                "train": f"{BASE}/{task}/train/0000.parquet",
                "validation": f"{BASE}/{task}/{validation_split}/0000.parquet",
            }
            for population, filename in expected_files.items():
                _fetch(sources[population], staging / filename)
                actual = parquet_sha256(staging / filename)
                expected = config["data"]["parquet_sha256"][task][population]
                if actual != expected:
                    raise ValueError(f"Downloaded {task}.{population} hash mismatch: {actual}")
            os.rename(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        print(f"Saved and verified {task}: {target}")
