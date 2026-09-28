from __future__ import annotations

import gzip
import json

import numpy as np

from task8_p01.assets.catalog import resolve_snapshot
from task8_p01.common.config import TASKS, Snapshot, resolve_path
from task8_p01.common.io import checked_complete, read_json, write_csv
from task8_p01.compat.task6 import load_task6_config
from task8_p01.data.panel import panel_members
from task8_p01.evaluation.runner import result_path


def _rows(path, key):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)[key]


def compare_diagnostic_endpoints(config, snapshot, run_id="p01"):
    """Compare new M00/M11 records to the matching old Task 6 full capture."""
    if not isinstance(snapshot, Snapshot):
        snapshot = Snapshot(**snapshot)
    task6_config = load_task6_config(config)
    from task6_phased.common.config import Condition, run_path

    _, entry = resolve_snapshot(config, snapshot)
    specifications = (
        ("M00", Condition(snapshot.experts, "R2", snapshot.k, None), "static"),
        ("M11", Condition(
            snapshot.experts, "R4d", snapshot.k,
            entry["canonical_source_seed"] if snapshot.role == "init" else snapshot.seed,
        ), entry["state_name"]),
    )
    rows = []
    for mode, condition, state_name in specifications:
        new_root = result_path(config, "diagnostic_128", snapshot, mode, entry["step"])
        checked_complete(new_root)
        old_root = run_path(
            task6_config, "capture/A", condition, config["protocol"]["source_run_id"]
        ) / state_name
        checked_complete(old_root)
        for task in TASKS:
            members, panel_hash = panel_members(config, task)
            wanted = set(members)
            old = [row for row in _rows(old_root / f"{task}.json.gz", "candidate") if row["source_index"] in wanted]
            new = _rows(new_root / f"{task}.json.gz", "predictions")
            old_by_id = {row["source_index"]: row for row in old}
            if len(old_by_id) != 128 or len(new) != 128:
                raise ValueError(f"{mode}/{task}: diagnostic subset is incomplete")
            ordered_old = [old_by_id[row["source_index"]] for row in new]
            gold_mismatch = sum(left["gold"] != right["gold"] for left, right in zip(ordered_old, new))
            prediction_mismatch = sum(left["prediction"] != right["prediction"] for left, right in zip(ordered_old, new))
            score_error = max(
                abs(float(left_score) - float(right_score))
                for left, right in zip(ordered_old, new)
                for left_score, right_score in zip(left["candidate_scores"], right["candidate_scores"])
            )
            # The old score was produced in a full-validation dynamic-padding
            # batch, while diagnostic_128 is regrouped into its own batches.
            # GPU GEMM shapes therefore differ even though masked examples are
            # semantically identical. Prediction/native identity is the P0
            # gate; 1e-2 is a separate diagnostic score warning threshold.
            semantic_status = "pass" if gold_mismatch == 0 and prediction_mismatch == 0 else "fail"
            score_status = "pass" if score_error <= 1.0e-2 else "warn"
            status = semantic_status
            rows.append({
                "E": snapshot.experts, "k": snapshot.k, "seed": "" if mode == "M00" else snapshot.seed,
                "checkpoint_role": "static" if mode == "M00" else snapshot.role,
                "checkpoint_step": 0 if mode == "M00" else entry["step"],
                "mode": mode, "task": task, "panel_hash": panel_hash,
                "count": len(new), "gold_mismatches": gold_mismatch,
                "prediction_mismatches": prediction_mismatch,
                "max_candidate_score_error": score_error,
                "score_diagnostic_tolerance": 1.0e-2, "score_status": score_status,
                "semantic_status": semantic_status, "status": status,
                "old_artifact": str(old_root), "new_artifact": str(new_root),
            })
    output = (
        resolve_path(config, config["execution"]["output_root"])
        / "results" / run_id / "tables"
        / (
            f"p0_endpoint_diagnostic_E{snapshot.experts}_k{snapshot.k}_init.csv"
            if snapshot.role == "init"
            else f"p0_endpoint_diagnostic_E{snapshot.experts}_k{snapshot.k}_best_seed{snapshot.seed}.csv"
        )
    )
    write_csv(output, rows)
    if any(row["status"] != "pass" for row in rows):
        raise AssertionError(f"Endpoint diagnostic comparison failed: {output}")
    return {"rows": len(rows), "status": "pass", "output": str(output)}
