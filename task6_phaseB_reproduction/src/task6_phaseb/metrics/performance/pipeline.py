from __future__ import annotations

from task6_phaseb.capture.storage import read_parquet
from task6_phaseb.common.config import protocol_id, recorded_protocol, run_path
from task6_phaseb.common.io import read_json, write_json
from task6_phaseb.imports.phase_a_f0 import baseline_accuracy
from task6_phaseb.metrics.performance.core import choose_best, performance


def captured_states(config, condition, run_id):
    base = run_path(config, "capture/validation", condition, run_id)
    values = []
    for path in base.glob("*/A/complete.json"):
        header = read_json(path)["header"]
        if header["protocol"] != recorded_protocol(config, condition, run_id) or header["condition"] != condition.to_dict():
            raise ValueError(f"Capture identity mismatch: {path}")
        values.append(header["state"])
    values.sort(key=lambda state: state["epoch"])
    expected = list(range(config["training"]["epochs"] + 1)) if condition.trainable else [0]
    if [state["epoch"] for state in values] != expected:
        raise ValueError(f"Missing/duplicate candidate A states: {base}")
    return values


def load_a(config, condition, run_id, state):
    from task6_phaseb.metrics.pipeline import check_source

    path, _ = check_source(config, condition, run_id, state, "A")
    records = read_parquet(path / "predictions.parquet")
    count = config["suite"].get("validation_limit", config["tasks"][condition.task]["validation_count"])
    return performance(records, expected_ids=list(range(count))), records


def compute_performance(config, condition, run_id):
    from task6_phaseb.metrics.pipeline import store_metric

    dense_accuracy = baseline_accuracy(config, run_id, condition.task)
    candidates = []
    for state in captured_states(config, condition, run_id):
        initial, records = load_a(config, condition, run_id, state)
        result = performance(records, dense_accuracy, expected_ids=list(range(initial["count"])))
        store_metric(config, condition, run_id, state, "performance", {"model": result, "layers": {}})
        candidates.append(dict(result, state=state))
    return candidates


def select_best(config, condition, run_id):
    candidates = compute_performance(config, condition, run_id)
    selected = choose_best(candidates)
    path = run_path(config, "metrics/performance", condition, run_id)
    result = {
        "protocol": protocol_id(config),
        "condition": condition.to_dict(),
        "selection": "best_validation",
        "state": selected["state"],
        "candidates": candidates,
    }
    write_json(path / "selection.json", result)
    return result


def best_state(config, condition, run_id):
    if not condition.trainable:
        return {"name": "static", "epoch": 0, "step": 0}
    selection = read_json(run_path(config, "metrics/performance", condition, run_id) / "selection.json")
    if selection["protocol"] != protocol_id(config) or selection["condition"] != condition.to_dict():
        raise ValueError("Best selection belongs to another condition/protocol")
    return selection["state"]
