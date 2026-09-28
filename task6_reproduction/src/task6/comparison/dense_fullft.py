"""Append a matched Dense-fullFT baseline to a completed routed-only result set."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from task6.aggregation.pipeline import aggregate_rows, collect_rows
from task6.common.config import ARMS, DENSE_FT_ARM, analysis_id, conditions, protocol_id, root_for
from task6.common.io import read_json, sha256, write_json


def _component(root, section, run_id):
    path = Path(root).resolve() / "data" / section / run_id / "metrics.json"
    return path, read_json(path)


def _display_name(arm):
    if arm == "dense":
        return "Dense-init"
    if arm == DENSE_FT_ARM:
        return "Dense-fullFT"
    return "G2-0.001" if arm == "G2" else arm


def _normalize_display(rows):
    result = []
    for source in rows:
        row = deepcopy(source)
        row["display_name"] = _display_name(row["arm"])
        result.append(row)
    return result


def _validate_sparse(rows, config):
    if any(row["arm"] == DENSE_FT_ARM for row in rows):
        raise ValueError("The routed source already contains Dense-fullFT")
    expected = {
        (task, variant["arm"], variant["name"], k)
        for task in config["suite"]["tasks"]
        for variant in config["variants"]
        for k in config["suite"]["top_k"]
    }
    observations = [
        (row["task"], row["arm"], row["variant"], row["k"])
        for row in rows
        if row["arm"] in ARMS and row["role"] == "best" and row["layer"] == "model"
        and row["group"] == "performance" and row["metric"] == "accuracy"
    ]
    if len(observations) != len(expected) or set(observations) != expected:
        raise ValueError("Routed source does not contain the complete 2×8×4 best-performance grid")
    for task in config["suite"]["tasks"]:
        references = [
            row for row in rows
            if row["task"] == task and row["arm"] == "dense" and row["role"] == "static"
            and row["layer"] == "model" and row["group"] == "performance"
            and row["metric"] == "relative_performance"
        ]
        if len(references) != 1 or abs(references[0]["mean"] - 100.0) > 1.0e-9:
            raise ValueError(f"Missing exact Dense-init 100% reference for {task}")


def report(config, dense_run_id, sparse_result_root, sparse_run_id):
    sources = {}
    for section in ("normalized", "aggregated", "paired_differences"):
        path, payload = _component(sparse_result_root, section, sparse_run_id)
        sources[section] = (path, payload)
    sparse_aggregated = _normalize_display(sources["aggregated"][1]["rows"])
    _validate_sparse(sparse_aggregated, config)

    dense_conditions = [condition for condition in conditions(config) if condition.arm == DENSE_FT_ARM]
    dense_normalized = collect_rows(config, dense_run_id, dense_conditions)
    dense_aggregated = aggregate_rows(dense_normalized, config)
    for task in config["suite"]["tasks"]:
        matches = [
            row for row in dense_aggregated
            if row["task"] == task and row["role"] == "best" and row["layer"] == "model"
            and row["group"] == "performance" and row["metric"] == "relative_performance"
        ]
        if len(matches) != 1:
            raise ValueError(f"Dense-fullFT best baseline is incomplete for {task}")

    meta = {
        "protocol": protocol_id(config), "analysis": analysis_id(config),
        "run_id": dense_run_id, "suite": "phase_a_f1_dense_fullft_extension",
        "sparse_run_id": sparse_run_id,
        "components": {
            section: {
                "path": str(path), "sha256": sha256(path),
                "meta": payload.get("meta", {}),
            }
            for section, (path, payload) in sources.items()
        },
        "selection_warning": "The validation set both selects and reports the best checkpoint.",
        "interpretation": "Dense-init remains 100%; Dense-fullFT is an additive matched-training baseline.",
    }
    destination = root_for(config) / "results/dense_fullft_extension" / dense_run_id
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing extension result: {destination}")
    sparse_root = Path(sparse_result_root).resolve()
    for kind in ("tables", "figures"):
        for section in ("main", "diagnostics", "appendix"):
            source = sparse_root / kind / section / sparse_run_id
            target = destination / kind / section / dense_run_id
            if not source.is_dir():
                raise FileNotFoundError(f"Completed routed-only rendering is missing: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
    write_json(destination / "data/normalized" / dense_run_id / "metrics.json", {
        "meta": meta,
        "rows": _normalize_display(sources["normalized"][1]["rows"]) + dense_normalized,
    })
    write_json(destination / "data/aggregated" / dense_run_id / "metrics.json", {
        "meta": meta, "rows": sparse_aggregated + dense_aggregated,
    })
    write_json(destination / "data/paired_differences" / dense_run_id / "metrics.json", {
        "meta": meta, "rows": sources["paired_differences"][1]["rows"],
    })

    from task6.visualization.render import figures, main_table
    main_table(config, dense_run_id, result_root=destination)
    figures(config, dense_run_id, result_root=destination,
            metrics={"accuracy", "relative_performance"})
    print(f"Dense-fullFT extension tables and figures: {destination}")
