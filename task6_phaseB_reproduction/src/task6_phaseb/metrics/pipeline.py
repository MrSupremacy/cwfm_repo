from __future__ import annotations

from itertools import zip_longest

import numpy as np

from task6_phaseb.aggregation.core import layer_summary
from task6_phaseb.capture.runner import capture_path
from task6_phaseb.capture.storage import probe_chunks, read_parquet, validate_selection
from task6_phaseb.common.config import analysis_id, protocol_id, recorded_protocol, run_path
from task6_phaseb.common.context import shared_path
from task6_phaseb.common.io import checked_complete, read_json, write_json
from task6_phaseb.metrics.load_balance.core import load_metrics
from task6_phaseb.metrics.selection_quality.core import selection_statistics
from task6_phaseb.metrics.stability.core import churn


METRICS = (
    "performance",
    "load_balance",
    "churn",
    "oracle_overlap",
    "adjusted_overlap",
    "activation_coverage",
    "oracle_resolution",
)
D_GROUPS = frozenset(("oracle_overlap", "adjusted_overlap", "activation_coverage", "oracle_resolution"))


def layer_names(config):
    return [
        f"{stack}_layer_{index:02d}"
        for stack in ("encoder", "decoder")
        for index in range(config["model"][f"{stack}_layers"])
    ]


def store_metric(config, condition, run_id, state, metric, values):
    path = run_path(config, f"metrics/{metric}", condition, run_id) / state["name"] / "metrics.json"
    write_json(path, {
        "protocol": protocol_id(config),
        "analysis": analysis_id(config),
        "condition": condition.to_dict(),
        "state": state,
        "metric_group": metric,
        **values,
    })


def check_source(config, condition, run_id, state, kind):
    path = capture_path(config, condition, run_id, state, kind)
    header = checked_complete(path)
    if (
        header["condition"] != condition.to_dict()
        or header["protocol"] != recorded_protocol(config, condition, run_id)
        or header["state"] != state
    ):
        raise ValueError(f"Capture identity mismatch: {path}")
    prepared = read_json(
        shared_path(config, "probe_sets", condition.task, condition.experts, run_id) / "context.json"
    )["header"]
    if header["input_header"] != prepared:
        raise ValueError("Capture input identity differs from prepared experiment")
    return path, header


def reduce_layers(layers, names, worst):
    model = {}
    for name in names:
        summary = layer_summary([value[name] for value in layers.values()], worst.get(name))
        model[name] = summary["mean"]
        model[f"{name}_layer_std"] = summary["layer_std"]
        if name in worst:
            model[f"{name}_worst"] = summary["worst"]
    return {"model": model, "layers": layers}


def compute_load(config, condition, run_id, state):
    path, _ = check_source(config, condition, run_id, state, "B")
    rows = read_parquet(path / "loads.parquet")
    if sorted(rows["layer_id"]) != sorted(layer_names(config)):
        raise ValueError("Incomplete/duplicate B layers")
    layers = {
        key: load_metrics(counts, tokens, condition.k)
        for key, tokens, counts in zip(
            rows["layer_id"], rows["valid_token_count"], rows["assignment_counts"]
        )
    }
    if any(len(counts) != condition.experts for counts in rows["assignment_counts"]):
        raise ValueError("B assignment width differs from condition E")
    store_metric(config, condition, run_id, state, "load_balance", reduce_layers(layers, ("cv",), {"cv": "max"}))


def compute_probe(config, condition, run_id, state, requested):
    path, header = check_source(config, condition, run_id, state, "probe")
    needed = requested & D_GROUPS
    if not needed:
        return
    if not header["with_q"]:
        return
    members = read_json(
        shared_path(config, "probe_sets", condition.task, condition.experts, run_id) / "members.json"
    )
    if sorted(item.name for item in path.iterdir() if item.is_dir() and item.name != "logs") != sorted(layer_names(config)):
        raise ValueError("Probe must contain exactly every configured layer")

    overlap_layers = {}
    adjusted_layers = {}
    coverage_layers = {}
    resolution_layers = {}
    for layer in layer_names(config):
        import hashlib

        key_hash = hashlib.sha256()
        count = positive_count = 0
        overlap_sum = adjusted_sum = coverage_sum = 0.0
        gap_sum = normalized_gap_sum = 0.0
        tie_count = zero_count = 0
        for block in probe_chunks(path / layer, config["metrics"]["chunk_rows"]):
            selected = block["selected_experts"].astype(np.int64)
            validate_selection(selected, condition.experts, condition.k)
            if not np.all(block["layer_id"] == layer):
                raise ValueError("Probe layer ID mismatch")
            keys = np.column_stack((block["sample_id"], block["token_position"]))
            key_hash.update(keys.astype("<i8").tobytes())
            q = block["expert_activation_sums"]
            if q.shape != (len(selected), condition.experts):
                raise ValueError("D q width differs from condition E")
            values = selection_statistics(selected, q)
            count += len(selected)
            overlap_sum += values["overlap"].sum(dtype=np.float64)
            adjusted_sum += values["adjusted_overlap"].sum(dtype=np.float64)
            coverage_sum += values["coverage"].sum(dtype=np.float64)
            positive_count += len(values["coverage"])
            gap_sum += values["boundary_gap"].sum(dtype=np.float64)
            normalized_gap_sum += values["normalized_boundary_gap"].sum(dtype=np.float64)
            tie_count += int(values["boundary_tie"].sum())
            zero_count += int(values["all_zero"].sum())
        stack = layer.split("_", 1)[0]
        if {"count": count, "sha256": key_hash.hexdigest()} != members["expected_keys"][stack]:
            raise ValueError(f"Probe keys do not cover fixed population: {layer}")
        overlap_layers[layer] = {"oracle_overlap": overlap_sum / count, "valid_token_count": count}
        adjusted_layers[layer] = {"adjusted_overlap": adjusted_sum / count, "valid_token_count": count}
        coverage_layers[layer] = {
            "activation_coverage": coverage_sum / positive_count if positive_count else None,
            "valid_token_count": positive_count,
            "zero_activation_count": zero_count,
        }
        resolution_layers[layer] = {
            "boundary_gap": gap_sum / count,
            "normalized_boundary_gap": normalized_gap_sum / positive_count if positive_count else None,
            "boundary_tie_rate": tie_count / count,
            "all_zero_rate": zero_count / count,
            "valid_token_count": count,
            "positive_activation_count": positive_count,
        }

    if "oracle_overlap" in needed:
        store_metric(config, condition, run_id, state, "oracle_overlap", reduce_layers(
            overlap_layers, ("oracle_overlap",), {"oracle_overlap": "min"}
        ))
    if "adjusted_overlap" in needed:
        store_metric(config, condition, run_id, state, "adjusted_overlap", reduce_layers(
            adjusted_layers, ("adjusted_overlap",), {"adjusted_overlap": "min"}
        ))
    if "activation_coverage" in needed:
        store_metric(config, condition, run_id, state, "activation_coverage", reduce_layers(
            coverage_layers, ("activation_coverage",), {"activation_coverage": "min"}
        ))
    if "oracle_resolution" in needed:
        store_metric(config, condition, run_id, state, "oracle_resolution", reduce_layers(
            resolution_layers,
            ("boundary_gap", "normalized_boundary_gap", "boundary_tie_rate", "all_zero_rate"),
            {"boundary_gap": "min", "normalized_boundary_gap": "min", "boundary_tie_rate": "max", "all_zero_rate": "max"},
        ))


def compute_churn(config, condition, run_id, previous, state):
    left_path, _ = check_source(config, condition, run_id, previous, "probe")
    right_path, _ = check_source(config, condition, run_id, state, "probe")
    members = read_json(
        shared_path(config, "probe_sets", condition.task, condition.experts, run_id) / "members.json"
    )
    layers = {}
    for layer in layer_names(config):
        import hashlib

        key_hash = hashlib.sha256()
        total = 0
        churn_sum = changed_sum = 0.0
        left = probe_chunks(left_path / layer, config["metrics"]["chunk_rows"])
        right = probe_chunks(right_path / layer, config["metrics"]["chunk_rows"])
        for first, second in zip_longest(left, right):
            if first is None or second is None:
                raise ValueError("Churn capture length mismatch")
            for key in ("sample_id", "token_position", "layer_id"):
                if not np.array_equal(first[key], second[key]):
                    raise ValueError("Churn token keys must align exactly")
            keys = np.column_stack((first["sample_id"], first["token_position"]))
            key_hash.update(keys.astype("<i8").tobytes())
            for block in (first, second):
                validate_selection(block["selected_experts"], condition.experts, condition.k)
                if not np.all(block["layer_id"] == layer):
                    raise ValueError("Churn layer mismatch")
            values, changed = churn(first["selected_experts"], second["selected_experts"])
            total += len(values)
            churn_sum += values.sum(dtype=np.float64)
            changed_sum += changed.sum(dtype=np.float64)
        if {"count": total, "sha256": key_hash.hexdigest()} != members["expected_keys"][layer.split("_")[0]]:
            raise ValueError("Churn population is incomplete")
        layers[layer] = {
            "churn": churn_sum / total,
            "exact_set_change": changed_sum / total,
            "valid_token_count": total,
        }
    values = reduce_layers(
        layers,
        ("churn", "exact_set_change"),
        {"churn": "max", "exact_set_change": "max"},
    )
    values["model"]["previous_state"] = previous
    store_metric(config, condition, run_id, state, "churn", values)


def compute_condition(config, condition, run_id, metric="all"):
    from task6_phaseb.metrics.performance.pipeline import best_state, captured_states, compute_performance

    requested = set(METRICS if metric == "all" else [metric])
    all_states = captured_states(config, condition, run_id)
    if "performance" in requested:
        compute_performance(config, condition, run_id)
    best = best_state(config, condition, run_id)["name"]
    for index, state in enumerate(all_states):
        if "load_balance" in requested and state["name"] in (best, "final", "static"):
            compute_load(config, condition, run_id, state)
        if requested & D_GROUPS:
            compute_probe(config, condition, run_id, state, requested)
        if "churn" in requested and condition.trainable and index:
            compute_churn(config, condition, run_id, all_states[index - 1], state)
