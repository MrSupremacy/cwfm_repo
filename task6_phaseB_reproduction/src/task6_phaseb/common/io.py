from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sys
import traceback
import uuid


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_files(root, paths):
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()


@contextmanager
def fresh_output(path):
    """No overwriting or partial-output reuse. A failed output remains for diagnosis."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    yield path


class Tee:
    def __init__(self, terminal, file, channel):
        self.terminal, self.file = terminal, file
        self.channel = channel

    def detach(self):
        # Third-party log handlers (and fork workers) may retain this proxy.
        # Release the per-run file before it closes, without closing the proxy.
        self.file = None

    def _fallback(self):
        current = getattr(sys, self.channel, None)
        return self.terminal if current is None or current is self else current

    def write(self, text):
        if self.file is None:
            return self._fallback().write(text)
        self.terminal.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        if self.file is None:
            self._fallback().flush()
            return
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return False


@contextmanager
def terminal_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = sys.stdout, sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        proxies = Tee(previous[0], stream, "stdout"), Tee(previous[1], stream, "stderr")
        sys.stdout, sys.stderr = proxies
        try:
            print("\n--- invocation ---")
            yield
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            for proxy in proxies:
                proxy.detach()
            sys.stdout, sys.stderr = previous


def complete(path, header):
    files = {p.relative_to(path).as_posix(): sha256(p) for p in Path(path).rglob("*")
             if p.is_file() and "logs" not in p.parts and p.name != "complete.json"}
    write_json(Path(path) / "complete.json", {"header": header, "files": files})


def checked_complete(path, expected=None):
    path = Path(path)
    result = read_json(path / "complete.json")
    if expected is not None and result["header"] != expected:
        raise ValueError(f"Output identity mismatch: {path}")
    for name, expected_hash in result["files"].items():
        if sha256(path / name) != expected_hash:
            raise ValueError(f"Output corruption: {path / name}")
    return result["header"]
