from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def path_identity(path, *, follow_symlinks=True):
    path = Path(path)
    resolved = path.resolve(strict=True) if follow_symlinks else path.absolute()
    files = sorted(item for item in resolved.rglob("*") if item.is_file()) if resolved.is_dir() else [resolved]
    return {
        "declared_path": str(path.absolute()),
        "resolved_path": str(resolved),
        "sha256": hashlib.sha256(json.dumps({
            item.relative_to(resolved).as_posix() if resolved.is_dir() else item.name: sha256(item)
            for item in files
        }, sort_keys=True).encode()).hexdigest(),
        "file_count": len(files),
    }


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    values = list(rows)
    if fieldnames is None:
        # Metric rows are intentionally heterogeneous (for example QQP adds
        # F1 while the other tasks only report accuracy).  Preserve the first
        # observed column order while accepting optional fields from later rows.
        names = []
        seen = set()
        for row in values:
            for name in row:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
    else:
        names = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)
    os.replace(temporary, path)


@contextmanager
def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    yield path


def complete(path, header):
    path = Path(path)
    files = {
        item.relative_to(path).as_posix(): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "complete.json"
    }
    write_json(path / "complete.json", {"header": header, "files": files})


def checked_complete(path, expected_header=None):
    path = Path(path)
    marker = read_json(path / "complete.json")
    if expected_header is not None and marker["header"] != expected_header:
        raise ValueError(f"Output identity mismatch: {path}")
    for name, expected in marker["files"].items():
        actual = sha256(path / name)
        if actual != expected:
            raise ValueError(f"Output content hash mismatch: {path / name}")
    return marker["header"]
