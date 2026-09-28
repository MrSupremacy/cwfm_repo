from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time
import uuid

import numpy as np

from task8_p01.common.io import checked_complete, complete, write_json
from task8_p2.progress import log


@contextmanager
def artifact_lock(path):
    """Advisory lock; kernel releases it on crash, avoiding stale PID lock files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    log(f"WAIT artifact lock {path}")
                    time.sleep(5)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def atomic_artifact(path, identity):
    """Publish only after all streams close and hashes are finalized; no overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    staging = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        yield staging
        write_json(staging / "identity.json", identity)
        complete(staging, identity)
        os.rename(staging, path)
    except BaseException:
        log(f"INCOMPLETE retained for inspection: {staging}")
        raise


def reuse(path, identity=None):
    path = Path(path)
    if not path.exists():
        return False
    checked_complete(path, identity)
    log(f"REUSE verified {path}")
    return True


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise TypeError(f"Object arrays are forbidden in caches: {key}")
    # NPZ is a portable, pickle-free container. Hidden tensors retain float32.
    np.savez(path, **arrays)


def read_npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name] for name in value.files}

