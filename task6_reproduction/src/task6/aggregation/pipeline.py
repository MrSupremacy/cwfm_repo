from __future__ import annotations

from collections import defaultdict

from task6.aggregation.core import seed_summary
from task6.common.config import analysis_id, conditions, protocol_id, root_for, run_path
from task6.common.io import read_json, write_json
from task6.metrics.performance.pipeline import best_state, captured_states
from task6.metrics.pipeline import layer_names


def required_groups(condition, state, best):
    if not condition.is_routed:
        return ["performance"]
    groups = ["performance", "oracle_overlap", "activation_coverage"]
    if state["name"] in {best, "final"}:
        groups.append("load_balance")
    if state["epoch"] > 0:
        groups.append("churn")
    return groups


def roles_for(condition, state, best, group):
    if condition.arm == "dense":
        return ["static"]
    roles = []
    if group != "churn" and state["name"] == best:
        roles.append("best")
    if state["name"] == "final":
        roles.append("final")
    if group in {"performance", "oracle_overlap", "activation_coverage", "churn"}:
        roles.append("trajectory")
    return roles


def collect_rows(config, run_id, selected=None):
    rows = []
    selected = conditions(config) if selected is None else selected
    for condition in selected:
        best = best_state(config, condition, run_id)["name"] if condition.trainable else "static"
        for state in captured_states(config, condition, run_id):
            for group in required_groups(condition, state, best):
                path = run_path(config, f"metrics/{group}", condition, run_id) / state["name"] / "metrics.json"
                item = read_json(path)
                if (item["condition"] != condition.to_dict() or item["protocol"] != protocol_id(config)
                        or item["state"] != state or item["analysis"] != analysis_id(config)):
                    raise ValueError(f"Metric identity mismatch: {path}")
                if group != "performance" and sorted(item["layers"]) != sorted(layer_names(config)):
                    raise ValueError(f"Metric layers incomplete: {path}")
                for layer, values in {"model": item["model"], **item["layers"]}.items():
                    for metric, value in values.items():
                        if isinstance(value, (list, dict)):
                            continue
                        for role in roles_for(condition, state, best, group):
                            rows.append({
                                **condition.to_dict(), "display_name": condition.display_name,
                                "state": state["name"], "epoch": state["epoch"], "step": state["step"],
                                "role": role, "group": group, "layer": layer,
                                "metric": metric, "value": value,
                            })
    return rows


def aggregate_rows(rows, config):
    grouped = defaultdict(list)
    fields = ("task", "arm", "variant", "display_name", "k", "role", "group", "layer", "metric")
    for row in rows:
        key = tuple(row[name] for name in fields) + (
            row["epoch"] if row["role"] == "trajectory" else None,)
        grouped[key].append(row)
    aggregated = []
    for key, items in grouped.items():
        identity = dict(zip((*fields, "epoch"), key))
        dense = identity["arm"] == "dense"
        expected = {None} if dense else set(config["suite"]["seeds"])
        if {row["seed"] for row in items} != expected or len(items) != len(expected):
            raise ValueError(f"Missing or duplicate seed observations: {identity}")
        aggregated.append({
            **identity, **seed_summary([row["value"] for row in items], dense),
            "states": [{name: row[name] for name in ("seed", "state", "epoch", "step")} for row in items],
        })
    return aggregated


def _meta(config, run_id):
    return {
        "protocol": protocol_id(config), "analysis": analysis_id(config),
        "run_id": run_id, "suite": config["suite"]["name"], "phase": "A",
        "regime": "F1_full_finetuning",
        "selection_warning": "The validation set both selects and reports the best checkpoint.",
    }


def paired_differences(rows, config):
    comparisons = (
        ("R4d", "R4o"), ("R4o", "G1"), ("R4d", "G1"),
        ("R4o", "G2"), ("R4d", "G2"),
    )
    output = []
    for task in config["suite"]["tasks"]:
        for k in config["suite"]["top_k"]:
            for role in ("best", "final"):
                pool = [row for row in rows if row["task"] == task and row["k"] == k
                        and row["role"] == role and row["layer"] == "model"]
                identities = sorted({(row["group"], row["metric"]) for row in pool})
                for reference, comparison in comparisons:
                    for group, metric in identities:
                        left = {row["seed"]: row["value"] for row in pool
                                if row["arm"] == reference and (row["group"], row["metric"]) == (group, metric)}
                        right = {row["seed"]: row["value"] for row in pool
                                 if row["arm"] == comparison and (row["group"], row["metric"]) == (group, metric)}
                        if not left and not right:
                            continue
                        expected = set(config["suite"]["seeds"])
                        if set(left) != expected or set(right) != expected:
                            raise ValueError(f"Incomplete paired comparison: {reference}-{comparison} {group}/{metric}")
                        values = [None if left[s] is None or right[s] is None else left[s] - right[s]
                                  for s in config["suite"]["seeds"]]
                        output.append({
                            "task": task, "k": k, "role": role, "group": group, "metric": metric,
                            "reference": reference, "comparison": comparison,
                            "direction": "reference_minus_comparison",
                            "seed_differences": dict(zip(map(str, config["suite"]["seeds"]), values)),
                            **seed_summary(values),
                        })
    return output


def aggregate(config, run_id):
    rows = collect_rows(config, run_id)
    aggregated = aggregate_rows(rows, config)
    meta = _meta(config, run_id)
    root = root_for(config) / "results/data"
    write_json(root / "normalized" / run_id / "metrics.json", {"meta": meta, "rows": rows})
    write_json(root / "aggregated" / run_id / "metrics.json", {"meta": meta, "rows": aggregated})
    write_json(root / "paired_differences" / run_id / "metrics.json",
               {"meta": meta, "rows": paired_differences(rows, config)})
    print(f"Aggregated {len(rows)} normalized rows into {len(aggregated)} seed summaries")
