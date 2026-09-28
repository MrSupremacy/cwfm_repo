from __future__ import annotations

from task6.capture.storage import read_parquet
from task6.common.config import Condition, protocol_id, run_path
from task6.common.io import read_json, write_json
from task6.metrics.performance.core import choose_best, performance
from task6.training.checkpoints import states


def captured_states(config, condition, run_id):
    """Return the complete scheduled state list and require an A capture for each."""
    expected = states(config, condition, run_id)
    base = run_path(config, "capture/validation", condition, run_id)
    actual = []
    for path in base.glob("*/A/complete.json"):
        header = read_json(path)["header"]
        if header["protocol"] != protocol_id(config) or header["condition"] != condition.to_dict():
            raise ValueError(f"Capture identity mismatch: {path}")
        actual.append(header["state"])
    actual.sort(key=lambda item: item["epoch"])
    if actual != expected:
        raise ValueError(f"Missing, duplicate, or unexpected A states: {base}")
    return expected


def load_a(config, condition, run_id, state):
    from task6.metrics.pipeline import check_source

    path, _ = check_source(config, condition, run_id, state, "A")
    records = read_parquet(path / "predictions.parquet")
    limit = config["suite"].get(
        "validation_limit", config["tasks"][condition.task]["validation_count"])
    return performance(records, expected_ids=list(range(limit))), records


def compute_performance(config, condition, run_id):
    from task6.metrics.pipeline import store_metric

    dense = Condition(condition.task, "dense")
    dense_result, _ = load_a(config, dense, run_id, captured_states(config, dense, run_id)[0])
    candidates = []
    for state in captured_states(config, condition, run_id):
        result, records = load_a(config, condition, run_id, state)
        result = performance(
            records, dense_result["accuracy"], expected_ids=list(range(result["count"])))
        store_metric(config, condition, run_id, state, "performance", {"model": result, "layers": {}})
        candidates.append({**result, "state": state})
    return candidates


def select_best(config, condition, run_id):
    candidates = compute_performance(config, condition, run_id)
    selected = choose_best(candidates)
    path = run_path(config, "metrics/performance", condition, run_id)
    result = {
        "protocol": protocol_id(config), "condition": condition.to_dict(),
        "selection": "max_correct_then_earliest_step", "state": selected["state"],
        "candidates": candidates,
    }
    write_json(path / "selection.json", result)
    return result


def best_state(config, condition, run_id):
    if condition.arm == "dense":
        return captured_states(config, condition, run_id)[0]
    selection = read_json(run_path(config, "metrics/performance", condition, run_id) / "selection.json")
    if selection["protocol"] != protocol_id(config) or selection["condition"] != condition.to_dict():
        raise ValueError("Best selection belongs to another condition/protocol")
    return selection["state"]
