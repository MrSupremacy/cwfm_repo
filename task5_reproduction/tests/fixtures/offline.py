"""Synthetic offline fixture. JSON table transport is mocked, never passed off as Parquet."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np

from task5.capture.runner import capture_path, header_for
from task5.common.config import R4_FAMILY, conditions, load_config, protocol_id, repository_root
from task5.common.context import input_header, shared_path
from task5.common.io import complete, read_json, write_json


def fixture_config(directory, phase_a=False):
    c = load_config(repository_root() / "configs/suites/phaseA_f0.yaml" if phase_a else None)
    c["suite"].update(name="fixture", tasks=["sst2"], top_k=[6], train_limit=4, validation_limit=4)
    c["execution"]["output_root"] = str(directory)
    c["model"].update(encoder_layers=1, decoder_layers=1, d_ff=128, expert_size=2)
    return c


def fixture_table_path(path):
    return Path(path).with_suffix(".fixture.json")


def mock_table_read(path):
    return read_json(fixture_table_path(path))


def mock_probe_chunks(path, chunk_rows=8192):
    data = read_json(Path(path) / "part.fixture.json")
    n = len(data["sample_id"])
    for start in range(0, n, chunk_rows):
        yield {key: np.asarray(values[start:start+chunk_rows]) for key, values in data.items()}


def build_fixture(config, run_id):
    names = ["encoder_layer_00", "decoder_layer_00"]
    E, k, n = 64, 6, 4
    ids = list(range(n))
    identity = {"dense": "synthetic-dense", "split": "synthetic-split", "data": "synthetic-data"}
    header = input_header(config, "sst2", identity)
    keys = np.column_stack((ids, np.zeros(n, dtype=np.int64))).astype("<i8")
    expected = {"count": n, "sha256": hashlib.sha256(keys.tobytes()).hexdigest()}
    prepared = shared_path(config, "probe_sets", "sst2", run_id)
    write_json(prepared / "context.json", {"header": header})
    write_json(prepared / "members.json", {"sample_ids": ids, "expected_keys": {"encoder": expected, "decoder": expected}})
    complete(prepared, header)
    static = shared_path(config, "static_routers", "sst2", run_id)
    static.mkdir(parents=True)
    np.savez(static / "labels.npz", **{key: np.arange(128, dtype=np.int64)//2 for key in names})
    complete(static, header)
    coact = shared_path(config, "coactivation", "sst2", run_id)
    coact.mkdir(parents=True)
    for layer in names:
        np.savez(coact / f"{layer}.npz", coactivation_sum=np.ones((128, 128), dtype=np.float32)*n, valid_token_count=np.int64(n))
    complete(coact, dict(header, kind="E", neuron_order="original", population="full_validation"))
    for condition in conditions(config):
        best_epoch = 5 if condition.arm == "R4" and condition.seed == 1 else 0
        for epoch in (range(11) if condition.trainable else [0]):
            name = ("final" if epoch == 10 else f"step_{epoch}") if condition.trainable else "static"
            state = {"name": name, "epoch": epoch, "step": epoch, "condition": condition.to_dict(), "protocol": protocol_id(config)}
            path = capture_path(config, condition, run_id, state, "A")
            correct = 4 if condition.arm == "dense" else (3 if epoch == best_epoch else 2)
            write_json(fixture_table_path(path / "predictions.parquet"), {"sample_id": ids,
                       "prediction": ["positive"] * n, "prediction_right": [i < correct for i in ids], "prediction_valid": [True] * n})
            complete(path, header_for(config, condition, state, header, "A"))
            if condition.arm == "dense":
                continue
            selected = np.tile((np.arange(k) + epoch + (condition.seed or 0)) % E, (n, 1))
            if not condition.trainable or epoch in (best_epoch, 10):
                path = capture_path(config, condition, run_id, state, "B")
                write_json(fixture_table_path(path / "loads.parquet"), {"layer_id": names, "valid_token_count": [n] * 2,
                           "assignment_counts": [np.bincount(selected.reshape(-1), minlength=E).tolist()] * 2})
                complete(path, header_for(config, condition, state, header, "B"))
            with_q = condition.arm in R4_FAMILY or not condition.trainable or epoch in (best_epoch, 10)
            path = capture_path(config, condition, run_id, state, "probe")
            for layer in names:
                values = {"sample_id": ids, "token_position": [0] * n, "layer_id": [layer] * n, "selected_experts": selected.tolist()}
                if with_q:
                    values["expert_activation_sums"] = np.tile(np.arange(E, 0, -1, dtype=np.float32), (n, 1)).tolist()
                write_json(path / layer / "part.fixture.json", values)
            complete(path, header_for(config, condition, state, header, "probe", with_q))
