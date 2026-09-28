from __future__ import annotations

from collections import defaultdict

from task6_phaseb.aggregation.core import seed_summary
from task6_phaseb.common.config import R4_FAMILY, analysis_id, conditions, protocol_id, root_for, run_path
from task6_phaseb.common.io import read_json, write_json
from task6_phaseb.imports.phase_a_f0 import load_import
from task6_phaseb.metrics.performance.pipeline import best_state, captured_states
from task6_phaseb.metrics.pipeline import D_GROUPS, layer_names


def required_groups(condition, state, best):
    groups = ["performance"]
    if state["name"] in (best, "final", "static"):
        groups.append("load_balance")
    if condition.arm in R4_FAMILY or state["name"] in (best, "final", "static"):
        groups.extend(sorted(D_GROUPS))
    if condition.trainable and state["epoch"] > 0:
        groups.append("churn")
    return groups


def roles_for(condition, state, best, group):
    if not condition.trainable:
        return ["static"]
    roles = []
    if group != "churn" and state["name"] == best:
        roles.append("best")
    if state["name"] == "final":
        roles.append("final")
    if group in ("performance", "churn") or (condition.arm in R4_FAMILY and group in D_GROUPS):
        roles.append("trajectory")
    return roles


def collect_new_rows(config, run_id):
    rows = []
    for condition in conditions(config):
        best = best_state(config, condition, run_id)["name"]
        for state in captured_states(config, condition, run_id):
            for group in required_groups(condition, state, best):
                path = run_path(config, f"metrics/{group}", condition, run_id) / state["name"] / "metrics.json"
                item = read_json(path)
                if (
                    item["condition"] != condition.to_dict()
                    or item["protocol"] != protocol_id(config)
                    or item["state"] != state
                    or item["analysis"] != analysis_id(config)
                ):
                    raise ValueError(f"Metric identity mismatch: {path}")
                if group != "performance" and sorted(item["layers"]) != sorted(layer_names(config)):
                    raise ValueError(f"Metric layers incomplete: {path}")
                for layer, values in {"model": item["model"], **item["layers"]}.items():
                    for metric, value in values.items():
                        if isinstance(value, (list, dict)):
                            continue
                        for role in roles_for(condition, state, best, group):
                            rows.append({
                                **condition.to_dict(),
                                "state": state["name"],
                                "epoch": state["epoch"],
                                "step": state["step"],
                                "role": role,
                                "group": group,
                                "layer": layer,
                                "metric": metric,
                                "value": value,
                                "source": "phase_b_new",
                            })
    return rows


def aggregate_rows(rows, config):
    groups = defaultdict(list)
    fields = ("task", "experts", "arm", "variant", "k", "role", "group", "layer", "metric")
    for row in rows:
        key = tuple(row[field] for field in fields)
        key += (row["epoch"] if row["role"] == "trajectory" else None,)
        groups[key].append(row)
    aggregated = []
    for key, items in groups.items():
        identity = dict(zip((*fields, "epoch"), key))
        deterministic = identity["arm"] in ("dense", "R2")
        expected = {None} if deterministic else set(config["suite"]["seeds"])
        seeds = {row["seed"] for row in items}
        if seeds != expected or len(items) != len(expected):
            raise ValueError(f"Missing/duplicate seed observations: {identity}; got={seeds}")
        aggregated.append({
            **identity,
            **seed_summary([row["value"] for row in items], deterministic),
            "states": [
                {field: row[field] for field in ("seed", "state", "epoch", "step")}
                for row in items
            ],
        })
    return aggregated


def paired_differences(rows, config):
    result = []
    candidates = ("R4o", "R4d", "G2-0.001", "G4")
    for task in config["suite"]["tasks"]:
        for experts, budgets in ((64, (6, 13, 19, 26)), (128, (13, 26, 38, 51)), (256, (26, 51, 77, 102))):
            for k in budgets:
                for role in ("best", "final"):
                    for metric, factor, unit in (
                        ("accuracy", 100, "percentage_points"),
                        ("relative_performance", 1, "relative_percentage_points"),
                    ):
                        pool = [
                            row for row in rows
                            if row["task"] == task and row["experts"] == experts and row["k"] == k
                            and row["group"] == "performance" and row["layer"] == "model"
                            and row["metric"] == metric and row["role"] == role
                        ]
                        reference = {row["seed"]: row for row in pool if row["arm"] == "G1"}
                        if set(reference) != set(config["suite"]["seeds"]):
                            raise ValueError(f"Missing G1 paired reference: {(task, experts, k, role, metric)}")
                        for arm in candidates:
                            comparison = {row["seed"]: row for row in pool if row["arm"] == arm}
                            if set(comparison) != set(reference):
                                raise ValueError(f"Missing paired arm {arm}: {(task, experts, k, role, metric)}")
                            differences = [
                                None if comparison[seed]["value"] is None or reference[seed]["value"] is None
                                else factor * (comparison[seed]["value"] - reference[seed]["value"])
                                for seed in config["suite"]["seeds"]
                            ]
                            result.append({
                                "task": task,
                                "experts": experts,
                                "k": k,
                                "role": role,
                                "reference": "G1",
                                "comparison": arm,
                                "metric": metric,
                                "unit": unit,
                                "seed_differences": dict(zip(map(str, config["suite"]["seeds"]), differences)),
                                **seed_summary(differences),
                            })
    return result


def gap_changes(paired, config):
    budgets = {
        "ratio10": {64: 6, 128: 13, 256: 26},
        "ratio20": {64: 13, 128: 26, 256: 51},
        "ratio30": {64: 19, 128: 38, 256: 77},
        "ratio40": {64: 26, 128: 51, 256: 102},
    }
    lookup = {
        (row["task"], row["experts"], row["k"], row["role"], row["comparison"], row["metric"]): row
        for row in paired
    }
    result = []
    for task in config["suite"]["tasks"]:
        for budget_name, mapping in budgets.items():
            for experts in (128, 256):
                for role in ("best", "final"):
                    for arm in ("R4o", "R4d", "G2-0.001", "G4"):
                        for metric, unit in (("accuracy", "percentage_points"), ("relative_performance", "relative_percentage_points")):
                            base = lookup[(task, 64, mapping[64], role, arm, metric)]
                            current = lookup[(task, experts, mapping[experts], role, arm, metric)]
                            values = [
                                None if base["seed_differences"][str(seed)] is None or current["seed_differences"][str(seed)] is None
                                else current["seed_differences"][str(seed)] - base["seed_differences"][str(seed)]
                                for seed in config["suite"]["seeds"]
                            ]
                            result.append({
                                "task": task,
                                "budget": budget_name,
                                "experts": experts,
                                "k": mapping[experts],
                                "baseline_experts": 64,
                                "baseline_k": mapping[64],
                                "role": role,
                                "reference": "G1",
                                "comparison": arm,
                                "metric": metric,
                                "unit": unit,
                                "seed_changes": dict(zip(map(str, config["suite"]["seeds"]), values)),
                                **seed_summary(values),
                            })
    return result


def aggregate(config, run_id):
    imported = load_import(config, run_id)["rows"]
    new = collect_new_rows(config, run_id)
    rows = [*imported, *new]
    aggregated = aggregate_rows(rows, config)
    paired = paired_differences(rows, config)
    changes = gap_changes(paired, config)
    meta = {
        "protocol": protocol_id(config),
        "analysis": analysis_id(config),
        "run_id": run_id,
        "suite": config["suite"]["name"],
        "selection_warning": "best-validation, not independent held-out test performance",
        "e64_source": "phase_a_f0_import",
    }
    root = root_for(config) / "results/data"
    write_json(root / "normalized" / run_id / "metrics.json", {"meta": meta, "rows": rows})
    write_json(root / "aggregated" / run_id / "metrics.json", {"meta": meta, "rows": aggregated})
    write_json(root / "paired_differences" / run_id / "metrics.json", {"meta": meta, "rows": paired})
    write_json(root / "gap_changes" / run_id / "metrics.json", {"meta": meta, "rows": changes})
    print(f"Aggregated {len(imported)} imported + {len(new)} new normalized rows")
