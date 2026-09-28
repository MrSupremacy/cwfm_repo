from __future__ import annotations

from itertools import zip_longest
import hashlib

import numpy as np

from task6.aggregation.core import layer_summary
from task6.capture.runner import capture_path
from task6.capture.storage import probe_chunks, read_parquet, validate_selection
from task6.common.config import METRICS, analysis_id, protocol_id, run_path
from task6.common.context import shared_path
from task6.common.io import checked_complete, read_json, write_json
from task6.metrics.load_balance.core import load_metrics
from task6.metrics.selection_quality.core import overlap_and_coverage
from task6.metrics.stability.core import churn


def layer_names(config):
    return [f"{stack}_layer_{index:02d}" for stack in ("encoder", "decoder")
            for index in range(config["model"][f"{stack}_layers"])]


def store_metric(config, condition, run_id, state, metric, values):
    path = run_path(config, f"metrics/{metric}", condition, run_id) / state["name"] / "metrics.json"
    write_json(path, {
        "protocol": protocol_id(config), "analysis": analysis_id(config),
        "condition": condition.to_dict(), "state": state, "metric_group": metric, **values,
    })


def check_source(config, condition, run_id, state, kind):
    path = capture_path(config, condition, run_id, state, kind)
    header = checked_complete(path)
    if (header["condition"] != condition.to_dict() or header["protocol"] != protocol_id(config)
            or header["state"] != state):
        raise ValueError(f"Capture identity mismatch: {path}")
    prepared = read_json(shared_path(config, "probe_sets", condition.task, run_id) / "context.json")["header"]
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
        raise ValueError("Incomplete or duplicate B layers")
    layers = {layer: load_metrics(counts, tokens, condition.k)
              for layer, tokens, counts in zip(
                  rows["layer_id"], rows["valid_token_count"], rows["assignment_counts"])}
    store_metric(config, condition, run_id, state, "load_balance",
                 reduce_layers(layers, ["cv"], {"cv": "max"}))


def _key_digest(block, digest):
    keys = np.column_stack((block["sample_id"], block["token_position"]))
    digest.update(keys.astype("<i8").tobytes())
    return len(keys)


def compute_probe(config, condition, run_id, state):
    path, header = check_source(config, condition, run_id, state, "probe")
    if not header.get("with_q"):
        raise ValueError("Phase A C+D capture must contain current local activation sums")
    members = read_json(shared_path(config, "probe_sets", condition.task, run_id) / "members.json")
    if sorted(p.name for p in path.iterdir() if p.is_dir() and p.name != "logs") != sorted(layer_names(config)):
        raise ValueError("Probe must contain exactly every configured layer")
    overlap_layers, coverage_layers = {}, {}
    for layer in layer_names(config):
        key_hash = hashlib.sha256()
        count = coverage_count = zero_count = 0
        overlap_sum = coverage_sum = 0.0
        for block in probe_chunks(path / layer, config["metrics"]["chunk_rows"]):
            selected = block["selected_experts"].astype(np.int64)
            validate_selection(selected, config["model"]["num_experts"], condition.k)
            if not np.all(block["layer_id"] == layer):
                raise ValueError("Probe layer ID mismatch")
            count += _key_digest(block, key_hash)
            overlap, coverage, zeros = overlap_and_coverage(
                selected, block["expert_activation_sums"])
            overlap_sum += overlap.sum(dtype=np.float64)
            coverage_sum += coverage.sum(dtype=np.float64)
            coverage_count += len(coverage)
            zero_count += zeros
        if {"count": count, "sha256": key_hash.hexdigest()} != members["expected_keys"][layer.split("_", 1)[0]]:
            raise ValueError(f"Probe population is incomplete: {layer}")
        overlap_layers[layer] = {
            "oracle_overlap": float(overlap_sum / count), "valid_token_count": count}
        coverage_layers[layer] = {
            "activation_coverage": float(coverage_sum / coverage_count) if coverage_count else None,
            "valid_token_count": coverage_count, "zero_activation_count": zero_count,
        }
    store_metric(config, condition, run_id, state, "oracle_overlap",
                 reduce_layers(overlap_layers, ["oracle_overlap"], {"oracle_overlap": "min"}))
    store_metric(config, condition, run_id, state, "activation_coverage",
                 reduce_layers(coverage_layers, ["activation_coverage"], {"activation_coverage": "min"}))


def compute_churn(config, condition, run_id, previous, state):
    left_path, _ = check_source(config, condition, run_id, previous, "probe")
    right_path, _ = check_source(config, condition, run_id, state, "probe")
    members = read_json(shared_path(config, "probe_sets", condition.task, run_id) / "members.json")
    layers = {}
    for layer in layer_names(config):
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
            total += _key_digest(first, key_hash)
            for block in (first, second):
                validate_selection(block["selected_experts"], config["model"]["num_experts"], condition.k)
                if not np.all(block["layer_id"] == layer):
                    raise ValueError("Churn layer mismatch")
            values, changed = churn(first["selected_experts"], second["selected_experts"])
            churn_sum += values.sum(dtype=np.float64)
            changed_sum += changed.sum(dtype=np.float64)
        if {"count": total, "sha256": key_hash.hexdigest()} != members["expected_keys"][layer.split("_", 1)[0]]:
            raise ValueError("Churn population is incomplete")
        layers[layer] = {
            "churn": float(churn_sum / total),
            "exact_set_change": float(changed_sum / total), "valid_token_count": total,
        }
    values = reduce_layers(layers, ["churn", "exact_set_change"],
                           {"churn": "max", "exact_set_change": "max"})
    values["previous_state"] = previous
    store_metric(config, condition, run_id, state, "churn", values)


def compute_condition(config, condition, run_id, metric="all"):
    from task6.metrics.performance.pipeline import best_state, captured_states, compute_performance

    requested = set(METRICS if metric == "all" else [metric])
    if not requested <= set(METRICS):
        raise ValueError(f"Unsupported metric request: {requested}")
    all_states = captured_states(config, condition, run_id)
    if "performance" in requested:
        compute_performance(config, condition, run_id)
    if not condition.is_routed:
        return
    best = best_state(config, condition, run_id)["name"]
    for index, state in enumerate(all_states):
        if "load_balance" in requested and state["name"] in {best, "final"}:
            compute_load(config, condition, run_id, state)
        if requested & {"oracle_overlap", "activation_coverage"}:
            # One pass writes both tightly coupled D-derived groups.
            compute_probe(config, condition, run_id, state)
        if "churn" in requested and index:
            compute_churn(config, condition, run_id, all_states[index - 1], state)
