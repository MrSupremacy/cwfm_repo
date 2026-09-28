from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def write_parquet(path, columns, kind, experts=0, k=0, compression_level=3):
    import pyarrow as pa
    import pyarrow.parquet as pq
    types = {"sample_id": pa.int64(), "token_position": pa.int32(), "prediction": pa.string(),
             "prediction_right": pa.bool_(), "prediction_valid": pa.bool_(), "layer_id": pa.string(),
             "valid_token_count": pa.int64()}
    arrays = {}
    for name, values in columns.items():
        if name in ("selected_experts", "expert_activation_sums", "assignment_counts"):
            dtype, width = {"selected_experts": (pa.uint8(), k), "expert_activation_sums": (pa.float32(), experts),
                            "assignment_counts": (pa.int64(), experts)}[name]
            a = np.asarray(values)
            if a.ndim != 2 or a.shape[1] != width:
                raise ValueError(f"Invalid capture shape: {name}")
            arrays[name] = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), type=dtype), width)
        else:
            arrays[name] = pa.array(values, type=types[name])
    table = pa.table(arrays).replace_schema_metadata({b"task6_phaseb_capture_kind": kind.encode()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=compression_level)


def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pydict()


def probe_chunks(path, chunk_rows=8192):
    import pyarrow.parquet as pq
    files = sorted(Path(path).glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No probe shards: {path}")
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=chunk_rows):
            # Lists are at most chunk_rows by E, never the full capture.
            yield {key: np.asarray(value) for key, value in batch.to_pydict().items()}


def validate_selection(selected, experts, k):
    s = np.asarray(selected)
    if s.ndim != 2 or s.shape[1] != k or not np.issubdtype(s.dtype, np.integer):
        raise ValueError("Invalid selection array")
    if np.any(s < 0) or np.any(s >= experts) or np.any(np.diff(np.sort(s.astype(np.int64), axis=1), axis=1) == 0):
        raise ValueError("Expert IDs must be distinct and in range")


class ProbeWriter:
    def __init__(self, directory, layer, expected, k, experts, with_q, shard_rows, compression_level):
        self.directory, self.layer, self.expected = Path(directory), layer, expected
        self.k, self.experts, self.with_q = k, experts, with_q
        self.shard_rows, self.compression_level = shard_rows, compression_level
        self.buffer = {}
        self.count = self.part = 0
        self.hash = hashlib.sha256()
        self.previous_key = None

    def add(self, keys, selected, q=None):
        keys = np.asarray(keys, dtype=np.int64)
        validate_selection(selected, self.experts, self.k)
        if keys.shape != (len(selected), 2):
            raise ValueError("Invalid probe key shape")
        if len(keys):
            previous = np.vstack(([self.previous_key], keys)) if self.previous_key is not None else keys
            bad = (previous[1:, 0] < previous[:-1, 0]) | ((previous[1:, 0] == previous[:-1, 0]) & (previous[1:, 1] <= previous[:-1, 1]))
            if np.any(bad):
                raise ValueError("Probe keys duplicated or out of sample/token order")
            self.previous_key = keys[-1].copy()
        self.hash.update(keys.astype("<i8").tobytes())
        self.count += len(keys)
        values = {"sample_id": keys[:, 0], "token_position": keys[:, 1].astype(np.int32),
                  "selected_experts": np.asarray(selected, dtype=np.uint8)}
        if self.with_q:
            if q is None or q.shape != (len(keys), self.experts) or not np.isfinite(q).all() or np.any(q < 0):
                raise ValueError("D requires all nonnegative finite expert activation sums")
            values["expert_activation_sums"] = q.astype(np.float32)
        for name, array in values.items():
            self.buffer[name] = np.concatenate((self.buffer[name], array)) if name in self.buffer else array.copy()
        while len(self.buffer["sample_id"]) >= self.shard_rows:
            self.flush(self.shard_rows)

    def flush(self, count):
        if not count:
            return
        columns = {name: array[:count] for name, array in self.buffer.items()}
        columns["layer_id"] = [self.layer] * count
        write_parquet(self.directory / f"part_{self.part:05d}.parquet", columns, "CD" if self.with_q else "C",
                      self.experts, self.k, self.compression_level)
        self.buffer = {name: array[count:] for name, array in self.buffer.items()}
        self.part += 1

    def finish(self):
        self.flush(len(self.buffer.get("sample_id", [])))
        actual = {"count": self.count, "sha256": self.hash.hexdigest()}
        if actual != self.expected:
            raise ValueError(f"Incomplete or misaligned probe {self.layer}: {actual} != {self.expected}")
