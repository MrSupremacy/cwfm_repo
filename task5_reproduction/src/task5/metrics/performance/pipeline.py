from __future__ import annotations

from task5.capture.runner import capture_path
from task5.capture.storage import read_parquet
from task5.common.config import Condition, protocol_id, recorded_protocol, run_path
from task5.common.io import checked_complete, read_json, write_json
from task5.metrics.performance.core import choose_best, performance


def captured_states(config, condition, run_id):
    base = run_path(config, "capture/validation", condition, run_id)
    values = []
    for path in base.glob("*/A/complete.json"):
        header = read_json(path)["header"]
        if header["protocol"] != recorded_protocol(config, condition, run_id) or header["condition"] != condition.to_dict():
            raise ValueError(f"Capture identity mismatch: {path}")
        values.append(header["state"])
    values.sort(key=lambda s: s["epoch"])
    expected = list(range(config["training"]["epochs"] + 1)) if condition.trainable else [0]
    if [s["epoch"] for s in values] != expected:
        raise ValueError(f"Missing/duplicate candidate A states: {base}")
    return values


def load_a(config, condition, run_id, state):
    from task5.metrics.pipeline import check_source
    path, _ = check_source(config, condition, run_id, state, "A")
    records = read_parquet(path / "predictions.parquet")
    n = config["suite"].get("validation_limit", config["tasks"][condition.task]["validation_count"])
    result = performance(records, expected_ids=list(range(n)))
    return result, records


def compute_performance(config, condition, run_id):
    from task5.metrics.pipeline import store_metric
    dense = Condition(condition.task, "dense")
    dense_state = captured_states(config, dense, run_id)[0]
    dense_result, _ = load_a(config, dense, run_id, dense_state)
    candidates = []
    for state in captured_states(config, condition, run_id):
        result, records = load_a(config, condition, run_id, state)
        result = performance(records, dense_result["accuracy"], expected_ids=list(range(result["count"])))
        store_metric(config, condition, run_id, state, "performance", {"model": result, "layers": {}})
        candidates.append(dict(result, state=state))
    return candidates


def select_best(config, condition, run_id):
    candidates = compute_performance(config, condition, run_id)
    selected = choose_best(candidates)
    path = run_path(config, "metrics/performance", condition, run_id)
    result = {"protocol": protocol_id(config), "condition": condition.to_dict(), "selection": "best_validation",
              "state": selected["state"], "candidates": candidates}
    write_json(path / "selection.json", result)
    return result


def best_state(config, condition, run_id):
    selection = read_json(run_path(config, "metrics/performance", condition, run_id) / "selection.json")
    if selection["protocol"] != protocol_id(config) or selection["condition"] != condition.to_dict():
        raise ValueError("Best selection belongs to another condition/protocol")
    return selection["state"]
