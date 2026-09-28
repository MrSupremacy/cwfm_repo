from __future__ import annotations

from functools import lru_cache
from itertools import zip_longest
import numpy as np

from task5.aggregation.core import layer_summary
from task5.capture.runner import capture_path
from task5.capture.storage import probe_chunks, read_parquet, validate_selection
from task5.common.config import analysis_id, digest, protocol_id, recorded_protocol, root_for, run_path
from task5.common.context import shared_path
from task5.common.io import checked_complete, read_json, write_json
from task5.metrics.load_balance.core import load_metrics
from task5.metrics.selection_quality.core import consistency_summary, expert_pair_matrix, overlap_and_coverage, pair_scores, random_reference
from task5.metrics.stability.core import churn

METRICS = ("performance", "load_balance", "churn", "oracle_overlap", "activation_coverage", "coactivation_consistency")


def layer_names(config):
    return [f"{s}_layer_{i:02d}" for s in ("encoder", "decoder") for i in range(config["model"][f"{s}_layers"])]


def store_metric(config, condition, run_id, state, metric, values):
    path = run_path(config, f"metrics/{metric}", condition, run_id) / state["name"] / "metrics.json"
    write_json(path, {"protocol": protocol_id(config), "analysis": analysis_id(config), "condition": condition.to_dict(), "state": state,
                      "metric_group": metric, **values})


def check_source(config, condition, run_id, state, kind):
    path = capture_path(config, condition, run_id, state, kind)
    header = checked_complete(path)
    if header["condition"] != condition.to_dict() or header["protocol"] != recorded_protocol(config, condition, run_id) or header["state"] != state:
        raise ValueError(f"Capture identity mismatch: {path}")
    prepared = read_json(shared_path(config, "probe_sets", condition.task, run_id) / "context.json")["header"]
    if header["input_header"] != prepared:
        raise ValueError("Capture input identity differs from prepared experiment")
    return path, header


def reduce_layers(layers, names, worst):
    out = {}
    for name in names:
        summary = layer_summary([value[name] for value in layers.values()], worst.get(name))
        out[name] = summary["mean"]
        out[f"{name}_layer_std"] = summary["layer_std"]
        if name in worst:
            out[f"{name}_worst"] = summary["worst"]
    return {"model": out, "layers": layers}


def compute_load(config, condition, run_id, state):
    path, _ = check_source(config, condition, run_id, state, "B")
    rows = read_parquet(path / "loads.parquet")
    if sorted(rows["layer_id"]) != sorted(layer_names(config)):
        raise ValueError("Incomplete/duplicate B layers")
    layers = {key: load_metrics(counts, tokens, condition.k) for key, tokens, counts in zip(
        rows["layer_id"], rows["valid_token_count"], rows["assignment_counts"])}
    names = ("cv", "gini", "maximum_share")
    store_metric(config, condition, run_id, state, "load_balance", reduce_layers(layers, names, dict.fromkeys(names, "max")))


@lru_cache(maxsize=32)
def coactivation_bundle(coact_path, static_path, experts):
    from pathlib import Path
    coact_path, static_path = Path(coact_path), Path(static_path)
    c_header = checked_complete(coact_path)
    s_header = checked_complete(static_path)
    if c_header["inputs"] != s_header["inputs"] or c_header["protocol"] != s_header["protocol"]:
        raise ValueError("Shared E and split use different input identities")
    result = {}
    with np.load(static_path / "labels.npz", allow_pickle=False) as labels:
        for key in labels.files:
            with np.load(coact_path / f"{key}.npz", allow_pickle=False) as raw:
                n = int(raw["valid_token_count"])
                result[key] = (expert_pair_matrix(raw["coactivation_sum"], n, labels[key], experts), n)
    identity = digest(read_json(coact_path / "complete.json"))
    return result, identity


def cached_random(config, task, run_id, layer, matrix, tokens, k, identity):
    m = config["metrics"]
    declaration = {"task": task, "layer": layer, "E": len(matrix), "k": k, "tokens": tokens,
                   "coactivation": identity, "seed": m["random_seed"], "repeats": m["random_repeats"],
                   "algorithm": m["random_algorithm"], "numpy": np.__version__}
    key = digest(declaration)
    path = root_for(config) / "runs/metrics/coactivation_consistency/random" / run_id / f"{key}.json"
    if path.exists():
        data = read_json(path)
        if data["declaration"] != declaration:
            raise ValueError("Random reference cache mismatch")
        return np.asarray(data["replicates"], dtype=np.float64)
    values = random_reference(matrix, tokens, k, m["random_repeats"], m["random_seed"], key, m["chunk_rows"])
    write_json(path, {"declaration": declaration, "replicates": values.tolist()})
    return values


def compute_probe(config, condition, run_id, state, requested):
    path, header = check_source(config, condition, run_id, state, "probe")
    need_d = header["with_q"] and bool({"oracle_overlap", "activation_coverage"} & requested)
    need_coact = "coactivation_consistency" in requested
    if not need_d and not need_coact:
        return
    members = read_json(shared_path(config, "probe_sets", condition.task, run_id) / "members.json")
    if sorted(p.name for p in path.iterdir() if p.is_dir() and p.name != "logs") != sorted(layer_names(config)):
        raise ValueError("Probe must contain exactly every configured layer")
    bundles, coact_identity = ({}, None)
    if need_coact:
        if checked_complete(shared_path(config, "static_routers", condition.task, run_id)) != header["input_header"]:
            raise ValueError("Probe and static split use different prepared inputs")
        bundles, coact_identity = coactivation_bundle(str(shared_path(config, "coactivation", condition.task, run_id)),
                                                      str(shared_path(config, "static_routers", condition.task, run_id)),
                                                      config["model"]["num_experts"])
    overlap_layers, coverage_layers, coact_layers = {}, {}, {}
    for layer in layer_names(config):
        import hashlib
        key_hash = hashlib.sha256()
        count = coverage_count = zero_count = 0
        overlap_sum = coverage_sum = selected_sum = 0.0
        for block in probe_chunks(path / layer, config["metrics"]["chunk_rows"]):
            s = block["selected_experts"].astype(np.int64)
            validate_selection(s, config["model"]["num_experts"], condition.k)
            if not np.all(block["layer_id"] == layer):
                raise ValueError("Probe layer ID mismatch")
            keys = np.column_stack((block["sample_id"], block["token_position"]))
            key_hash.update(keys.astype("<i8").tobytes())
            count += len(s)
            if need_d:
                overlap, coverage, zeros = overlap_and_coverage(s, block["expert_activation_sums"])
                overlap_sum += overlap.sum(dtype=np.float64)
                coverage_sum += coverage.sum(dtype=np.float64)
                coverage_count += len(coverage)
                zero_count += zeros
            if need_coact:
                selected_sum += pair_scores(s, bundles[layer][0]).sum(dtype=np.float64)
        stack = layer.split("_", 1)[0]
        if {"count": count, "sha256": key_hash.hexdigest()} != members["expected_keys"][stack]:
            raise ValueError(f"Probe keys do not cover fixed population: {layer}")
        if need_d:
            overlap_layers[layer] = {"oracle_overlap": float(overlap_sum / count), "valid_token_count": count}
            coverage_layers[layer] = {"activation_coverage": float(coverage_sum / coverage_count) if coverage_count else None,
                                      "valid_token_count": coverage_count, "zero_activation_count": zero_count}
        if need_coact:
            repeats = cached_random(config, condition.task, run_id, layer, bundles[layer][0], count, condition.k, coact_identity)
            coact_layers[layer] = dict(consistency_summary(selected_sum / count, repeats, config["metrics"]["random_ratio_epsilon"]),
                                       valid_token_count=count, reference_token_count=bundles[layer][1])
    if need_d:
        if "oracle_overlap" in requested:
            store_metric(config, condition, run_id, state, "oracle_overlap", reduce_layers(overlap_layers, ["oracle_overlap"], {"oracle_overlap": "min"}))
        if "activation_coverage" in requested:
            store_metric(config, condition, run_id, state, "activation_coverage", reduce_layers(coverage_layers, ["activation_coverage"], {"activation_coverage": "min"}))
    if need_coact:
        selected = np.mean([v["selected_mean"] for v in coact_layers.values()])
        replicates = np.mean([v["random_replicates"] for v in coact_layers.values()], axis=0)
        model = consistency_summary(selected, replicates, config["metrics"]["random_ratio_epsilon"])
        for key in ("selected_mean", "random_mean", "excess", "ratio"):
            model[f"{key}_layer_std"] = layer_summary([v[key] for v in coact_layers.values()])["layer_std"]
        store_metric(config, condition, run_id, state, "coactivation_consistency", {"model": model, "layers": coact_layers})


def compute_churn(config, condition, run_id, previous, state):
    a, _ = check_source(config, condition, run_id, previous, "probe")
    b, _ = check_source(config, condition, run_id, state, "probe")
    members = read_json(shared_path(config, "probe_sets", condition.task, run_id) / "members.json")
    layers = {}
    for layer in layer_names(config):
        import hashlib
        key_hash = hashlib.sha256()
        total = 0
        churn_sum = changed_sum = 0.0
        left, right = (probe_chunks(p / layer, config["metrics"]["chunk_rows"]) for p in (a, b))
        for x, y in zip_longest(left, right):
            if x is None or y is None:
                raise ValueError("Churn capture length mismatch")
            for key in ("sample_id", "token_position", "layer_id"):
                if not np.array_equal(x[key], y[key]):
                    raise ValueError("Churn token keys must align exactly, not by an inner join")
            keys = np.column_stack((x["sample_id"], x["token_position"]))
            key_hash.update(keys.astype("<i8").tobytes())
            for block in (x, y):
                validate_selection(block["selected_experts"], config["model"]["num_experts"], condition.k)
                if not np.all(block["layer_id"] == layer):
                    raise ValueError("Churn layer mismatch")
            values, changed = churn(x["selected_experts"], y["selected_experts"])
            total += len(values)
            churn_sum += values.sum(dtype=np.float64)
            changed_sum += changed.sum(dtype=np.float64)
        if {"count": total, "sha256": key_hash.hexdigest()} != members["expected_keys"][layer.split("_")[0]]:
            raise ValueError("Churn population is incomplete")
        layers[layer] = {"churn": float(churn_sum / total), "exact_set_change": float(changed_sum / total), "valid_token_count": total}
    values = reduce_layers(layers, ["churn", "exact_set_change"], {"churn": "max", "exact_set_change": "max"})
    values["previous_state"] = previous
    store_metric(config, condition, run_id, state, "churn", values)


def compute_condition(config, condition, run_id, metric="all"):
    from task5.metrics.performance.pipeline import best_state, captured_states, compute_performance
    requested = set(METRICS if metric == "all" else [metric])
    all_states = captured_states(config, condition, run_id)
    if "performance" in requested:
        compute_performance(config, condition, run_id)
    if condition.arm == "dense":
        return
    best = best_state(config, condition, run_id)["name"] if condition.trainable else "static"
    for i, state in enumerate(all_states):
        if "load_balance" in requested and state["name"] in (best, "final", "static"):
            compute_load(config, condition, run_id, state)
        if requested & {"oracle_overlap", "activation_coverage", "coactivation_consistency"}:
            compute_probe(config, condition, run_id, state, requested)
        if "churn" in requested and condition.trainable and i:
            compute_churn(config, condition, run_id, all_states[i - 1], state)
