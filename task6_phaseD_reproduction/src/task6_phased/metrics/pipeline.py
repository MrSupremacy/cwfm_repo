from __future__ import annotations

from itertools import zip_longest
import numpy as np

from task6_phased.capture.runner import capture_path
from task6_phased.capture.storage import probe_chunks, read_parquet, validate_selection
from task6_phased.common.config import TASKS, analysis_id, protocol_id, run_path
from task6_phased.common.context import verify_prepared
from task6_phased.common.io import checked_complete, write_json
from task6_phased.metrics.load_balance.core import load_metrics
from task6_phased.metrics.selection_quality.core import selection_statistics
from task6_phased.metrics.specialization.core import domain_metrics, utilization
from task6_phased.metrics.stability.core import churn
from task6_phased.training.checkpoints import states


METRICS = ("performance", "load_balance", "churn", "oracle_overlap", "activation_coverage", "specialization")


def layer_names(config):
    return [
        f"{stack}_layer_{index:02d}"
        for stack in ("encoder", "decoder")
        for index in range(config["model"][f"{stack}_layers"])
    ]


def _store(config, condition, run_id, state, group, payload):
    write_json(
        run_path(config, f"metrics/{group}", condition, run_id) / state["name"] / "metrics.json",
        {
            "protocol": protocol_id(config), "analysis": analysis_id(config),
            "condition": condition.to_dict(), "state": state,
            "metric_group": group, **payload,
        },
    )


def _source(config, condition, run_id, state, kind, task):
    path = capture_path(config, condition, run_id, state, kind, task)
    expected = {
        "schema": 2, "protocol": protocol_id(config), "condition": condition.to_dict(),
        "state": state, "input_header": verify_prepared(config, run_id, condition.experts),
        "kind": kind, "task": task, "with_q": kind == "probe",
    }
    checked_complete(path, expected)
    return path


def _model_layer_mean(layers, field, stack=None):
    values = [
        row[field] for name, row in layers.items()
        if stack is None or name.startswith(stack + "_")
    ]
    valid = [value for value in values if value is not None]
    return None if not valid else float(np.mean(valid))


def compute_load(config, condition, run_id, state):
    tasks, pooled = {}, {
        layer: np.zeros(condition.experts, dtype=np.int64)
        for layer in layer_names(config)
    }
    for task in TASKS:
        rows = read_parquet(_source(config, condition, run_id, state, "B", task) / "loads.parquet")
        layers = {}
        for layer, tokens, counts in zip(rows["layer_id"], rows["valid_token_count"], rows["assignment_counts"]):
            layers[layer] = load_metrics(counts, tokens, condition.k)
            pooled[layer] += np.asarray(counts, dtype=np.int64)
        tasks[task] = {
            "layers": layers,
            "encoder_mean": _model_layer_mean(layers, "cv", "encoder"),
            "decoder_mean": _model_layer_mean(layers, "cv", "decoder"),
            "all_layer_mean": _model_layer_mean(layers, "cv"),
        }
    global_layers = {
        layer: load_metrics(counts, int(counts.sum() // condition.k), condition.k)
        for layer, counts in pooled.items()
    }
    _store(config, condition, run_id, state, "load_balance", {
        "tasks": tasks, "equal_domain_global": {
            "layers": global_layers,
            "encoder_mean": _model_layer_mean(global_layers, "cv", "encoder"),
            "decoder_mean": _model_layer_mean(global_layers, "cv", "decoder"),
            "all_layer_mean": _model_layer_mean(global_layers, "cv"),
        },
    })


def _probe_arrays(path, layer, condition, population):
    selected, q, keys = [], [], []
    for block in probe_chunks(path / layer):
        mask = np.ones(len(block["sample_id"]), dtype=bool)
        if population == "encoder_content":
            mask = np.asarray(block["is_content"], dtype=bool)
        elif population == "encoder_prefix":
            mask = ~np.asarray(block["is_content"], dtype=bool) & ~np.asarray(block["is_special"], dtype=bool)
        selected.append(np.asarray(block["selected_experts"])[mask])
        if "expert_activation_sums" in block:
            q.append(np.asarray(block["expert_activation_sums"])[mask])
        keys.append(np.column_stack((block["sample_id"], block["token_position"]))[mask])
    selected = np.concatenate(selected) if selected else np.empty((0, condition.k), dtype=np.int64)
    validate_selection(selected, condition.experts, condition.k)
    return selected, (np.concatenate(q) if q else None), (np.concatenate(keys) if keys else np.empty((0, 2), dtype=np.int64))


def compute_probe_metrics(config, condition, run_id, state):
    quality, populations = {}, ("encoder_content", "encoder_prefix", "encoder_all", "decoder")
    distributions = {population: {} for population in populations}
    for task in TASKS:
        path = _source(config, condition, run_id, state, "probe", task)
        quality[task] = {"layers": {}}
        for layer in layer_names(config):
            population = "decoder" if layer.startswith("decoder") else "encoder_all"
            selected, q, _ = _probe_arrays(path, layer, condition, population)
            values = selection_statistics(selected, q)
            quality[task]["layers"][layer] = {
                "oracle_overlap": float(values["overlap"].mean()),
                "activation_coverage": None if len(values["coverage"]) == 0 else float(values["coverage"].mean()),
                "all_zero_count": int(values["all_zero"].sum()), "token_count": len(selected),
            }
            for target_population in (
                ("decoder",) if layer.startswith("decoder")
                else ("encoder_content", "encoder_prefix", "encoder_all")
            ):
                chosen, _, _ = _probe_arrays(path, layer, condition, target_population)
                distribution, counts = utilization(chosen, condition.experts)
                distributions[target_population].setdefault(layer, {})[task] = {
                    "distribution": distribution, "counts": counts,
                }
        quality[task]["all_layer_mean"] = {
            field: _model_layer_mean(quality[task]["layers"], field)
            for field in ("oracle_overlap", "activation_coverage")
        }
    specialization = {}
    for population, by_layer in distributions.items():
        specialization[population] = {"layers": {}}
        for layer, by_task in by_layer.items():
            values = domain_metrics(
                {task: by_task[task]["distribution"] for task in TASKS}, TASKS
            )
            specialization[population]["layers"][layer] = {
                **values,
                "domain_counts": {task: by_task[task]["counts"].tolist() for task in TASKS},
            }
        for field in ("mean_js_distance", "max_js_distance", "mi_bits", "nmi_domain"):
            specialization[population][f"{field}_layer_mean"] = _model_layer_mean(
                specialization[population]["layers"], field
            )
    _store(config, condition, run_id, state, "selection_quality", {"tasks": quality})
    _store(config, condition, run_id, state, "specialization", {"populations": specialization})


def compute_churn(config, condition, run_id, previous, state):
    tasks = {}
    for task in TASKS:
        left = _source(config, condition, run_id, previous, "probe", task)
        right = _source(config, condition, run_id, state, "probe", task)
        layers = {}
        for layer in layer_names(config):
            total = value_sum = 0
            left_chunks = probe_chunks(left / layer, config["metrics"]["chunk_rows"])
            right_chunks = probe_chunks(right / layer, config["metrics"]["chunk_rows"])
            for first, second in zip_longest(left_chunks, right_chunks):
                if first is None or second is None:
                    raise ValueError("Churn capture length mismatch")
                for key in ("sample_id", "token_position", "is_content", "is_special"):
                    if not np.array_equal(first[key], second[key]):
                        raise ValueError("Churn population/key mismatch")
                value, _ = churn(first["selected_experts"], second["selected_experts"])
                total += len(value)
                value_sum += float(value.sum())
            layers[layer] = {"churn": value_sum / total, "token_count": total}
        tasks[task] = {"layers": layers, "all_layer_mean": _model_layer_mean(layers, "churn")}
    _store(config, condition, run_id, state, "churn", {"previous_state": previous, "tasks": tasks})


def compute_condition(config, condition, run_id, metric="all"):
    from task6_phased.metrics.performance.pipeline import best_state, compute_performance
    requested = set(METRICS if metric == "all" else [metric])
    all_states = states(config, condition, run_id)
    if "performance" in requested:
        compute_performance(config, condition, run_id)
    selected_best = best_state(config, condition, run_id)["name"]
    final = f"step_{config['training']['total_optimizer_steps']}" if condition.trainable else "static"
    for index, state in enumerate(all_states):
        if "load_balance" in requested and state["name"] in {selected_best, final, "static"}:
            compute_load(config, condition, run_id, state)
        if requested & {"oracle_overlap", "activation_coverage", "specialization"}:
            compute_probe_metrics(config, condition, run_id, state)
        if "churn" in requested and condition.trainable and index:
            compute_churn(config, condition, run_id, all_states[index - 1], state)
