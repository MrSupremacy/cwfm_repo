from __future__ import annotations

import gzip
import json

from task6_phased.capture.runner import capture_path
from task6_phased.common.config import TASKS, analysis_id, protocol_id, run_path
from task6_phased.common.io import checked_complete, read_json, write_json
from task6_phased.training.checkpoints import states


def captured_states(config, condition, run_id):
    result = states(config, condition, run_id)
    for state in result:
        checked_complete(
            capture_path(config, condition, run_id, state, "A"),
            {
                "schema": 2, "protocol": protocol_id(config), "condition": condition.to_dict(),
                "state": state,
                "input_header": read_json(
                    run_path(config, "train", condition, run_id) / "config.json"
                )["input_header"] if condition.trainable else checked_complete(
                    __import__("task6_phased.common.context", fromlist=["artifact_path"]).artifact_path(
                        config, "static_routers", run_id, condition.experts
                    )
                ),
                "kind": "A", "task": None, "with_q": False,
            },
        )
    return result


def _a(config, condition, run_id, state):
    path = capture_path(config, condition, run_id, state, "A")
    summary = read_json(path / "summary.json")
    tasks = {}
    for task in TASKS:
        with gzip.open(path / f"{task}.json.gz", "rt", encoding="utf-8") as stream:
            tasks[task] = json.load(stream)["metrics"]
    return {"summary": summary, "tasks": tasks}


def compute_performance(config, condition, run_id):
    candidates = []
    for state in states(config, condition, run_id):
        value = _a(config, condition, run_id, state)
        output = {
            "protocol": protocol_id(config), "analysis": analysis_id(config),
            "condition": condition.to_dict(), "state": state,
            "metric_group": "performance", **value,
        }
        write_json(
            run_path(config, "metrics/performance", condition, run_id) / state["name"] / "metrics.json",
            output,
        )
        candidates.append({
            "state": state, "macro": value["summary"]["macro"],
            "worst_domain": value["summary"]["worst_domain"],
        })
    selected = min(candidates, key=lambda row: (-row["macro"], row["state"]["step"]))
    selection = {
        "protocol": protocol_id(config), "condition": condition.to_dict(),
        "selection": "native_macro_equal_domain_best", "tie_break": "earlier_step",
        "state": selected["state"], "candidates": candidates,
    }
    write_json(run_path(config, "metrics/performance", condition, run_id) / "selection.json", selection)
    return candidates


def best_state(config, condition, run_id):
    if not condition.trainable:
        return {"name": "static", "step": 0, "condition": condition.to_dict(), "protocol": protocol_id(config)}
    selection = read_json(run_path(config, "metrics/performance", condition, run_id) / "selection.json")
    if selection["protocol"] != protocol_id(config) or selection["condition"] != condition.to_dict():
        raise ValueError("Best selection identity mismatch")
    return selection["state"]


def select_best(config, condition, run_id):
    compute_performance(config, condition, run_id)
    return best_state(config, condition, run_id)
