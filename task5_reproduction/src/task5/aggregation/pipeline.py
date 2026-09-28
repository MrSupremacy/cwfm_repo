from __future__ import annotations

from collections import defaultdict

from task5.aggregation.core import seed_summary
from task5.common.config import R4_FAMILY, analysis_id, conditions, extension_spec, protocol_id, root_for, run_path
from task5.common.io import read_json, write_json
from task5.metrics.performance.pipeline import best_state, captured_states
from task5.metrics.pipeline import layer_names


def required_groups(condition, state, best):
    groups = ["performance"]
    if condition.arm == "dense":
        return groups
    groups.append("coactivation_consistency")
    if state["name"] in (best, "final", "static"):
        groups.append("load_balance")
    if condition.arm in R4_FAMILY or state["name"] in (best, "final", "static"):
        groups.extend(("oracle_overlap", "activation_coverage"))
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
    if group in ("performance", "coactivation_consistency", "churn") or (
            condition.arm in R4_FAMILY and group in ("oracle_overlap", "activation_coverage")):
        roles.append("trajectory")
    return roles


def collect_rows(config, run_id, selected=None):
    rows = []
    for condition in conditions(config) if selected is None else selected:
        best = best_state(config, condition, run_id)["name"] if condition.trainable else "static"
        for state in captured_states(config, condition, run_id):
            for group in required_groups(condition, state, best):
                path = run_path(config, f"metrics/{group}", condition, run_id) / state["name"] / "metrics.json"
                item = read_json(path)  # Missing required metrics fail, not silently disappear.
                if item["condition"] != condition.to_dict() or item["protocol"] != protocol_id(config) or item["state"] != state:
                    raise ValueError(f"Metric identity mismatch: {path}")
                if item["analysis"] != analysis_id(config):
                    raise ValueError("Metric implementation/config changed; recompute offline metrics before aggregation")
                if group != "performance" and sorted(item["layers"]) != sorted(layer_names(config)):
                    raise ValueError("Metric layers incomplete")
                for layer, values in {"model": item["model"], **item["layers"]}.items():
                    for metric, value in values.items():
                        if isinstance(value, (list, dict)):
                            continue  # Random replicate vectors remain in their metric JSONs.
                        for role in roles_for(condition, state, best, group):
                            rows.append({**condition.to_dict(), "state": state["name"], "epoch": state["epoch"], "step": state["step"],
                                         "role": role, "group": group, "layer": layer, "metric": metric, "value": value})
    return rows


def aggregate_rows(rows, config):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("task", "arm", "variant", "k", "role", "group", "layer", "metric"))
        key += (row["epoch"] if row["role"] == "trajectory" else None,)
        groups[key].append(row)
    aggregated = []
    for key, items in groups.items():
        identity = dict(zip(("task", "arm", "variant", "k", "role", "group", "layer", "metric", "epoch"), key))
        deterministic = identity["arm"] in ("dense", "R1", "R2", "R2-soft", "R3")
        expected = {None} if deterministic else set(config["suite"]["seeds"])
        if {row["seed"] for row in items} != expected or len(items) != len(expected):
            raise ValueError(f"Missing/duplicate seed observations: {identity}")
        aggregated.append({**identity, **seed_summary([row["value"] for row in items], deterministic),
                           "states": [{k: row[k] for k in ("seed", "state", "epoch", "step")} for row in items]})
    return aggregated


def _meta(config, run_id, extension_base=None):
    result = {"protocol": protocol_id(config), "analysis": analysis_id(config), "run_id": run_id, "suite": config["suite"]["name"],
            "selection_warning": "best-validation, not independent held-out test performance",
            "random_interval_warning": "Random-reference quantiles are not training-seed confidence intervals"}
    if extension_base is not None:
        result["extension_base"] = extension_base
    return result


def _base_snapshot(config, run_id, extension):
    """Preserve the completed base summaries before writing a combined view."""
    root = root_for(config) / "results/data"
    files = (("normalized", "metrics.json"), ("aggregated", "metrics.json"),
             ("aggregated", "paired_differences.json"))
    snapshot_root = root / "extension_base" / run_id
    loaded = {}
    for section, name in files:
        destination = snapshot_root / section / name
        if destination.exists():
            item = read_json(destination)
        else:
            source = root / section / run_id / name
            item = read_json(source)
            meta = item.get("meta", {})
            if meta.get("protocol") != extension["base_protocol"]:
                raise ValueError(f"Base result protocol mismatch; refusing to snapshot {source}")
            write_json(destination, item)
        meta = item.get("meta", {})
        if (meta.get("protocol"), meta.get("run_id"), meta.get("suite")) != (
                extension["base_protocol"], run_id, config["suite"]["name"]):
            raise ValueError(f"Invalid extension base snapshot: {destination}")
        loaded[(section, name)] = item
    return loaded


def aggregate(config, run_id):
    extension = extension_spec(config, run_id)
    if extension and "arms" in extension:
        raise ValueError("Phase A F0 must use phase-a-report; existing results are read-only")
    extension_base = None
    if extension is None:
        rows = collect_rows(config, run_id)
    else:
        snapshots = _base_snapshot(config, run_id, extension)
        base = snapshots[("normalized", "metrics.json")]
        if any(row["arm"] == extension["arm"] for row in base["rows"]):
            raise ValueError("Base snapshot already contains the extension arm")
        selected = [condition for condition in conditions(config) if condition.arm == extension["arm"]]
        added = collect_rows(config, run_id, selected)
        if not added or any(row["arm"] != extension["arm"] for row in added):
            raise ValueError("Extension aggregation selected an invalid condition set")
        rows = [*base["rows"], *added]
        extension_base = {"protocol": extension["base_protocol"], "analysis": base["meta"].get("analysis"),
                          "snapshot": f"results/data/extension_base/{run_id}"}

    aggregated = aggregate_rows(rows, config)
    result = root_for(config) / "results/data"
    meta = _meta(config, run_id, extension_base)
    write_json(result / "normalized" / run_id / "metrics.json", {"meta": meta, "rows": rows})
    write_json(result / "aggregated" / run_id / "metrics.json", {"meta": meta, "rows": aggregated})
    write_json(result / "aggregated" / run_id / "paired_differences.json", {"meta": meta, "rows": paired_differences(rows, config)})
    print(f"Aggregated {len(rows)} normalized rows into {len(aggregated)} rows")


def paired_differences(rows, config):
    result = []
    for task in config["suite"]["tasks"]:
        for k in config["suite"]["top_k"]:
            for role in ("best", "final"):
                for metric, factor, unit in (("accuracy", 100, "percentage_points"), ("relative_performance", 1, "relative_percentage_points")):
                    pool = [r for r in rows if r["task"] == task and r["k"] == k and r["group"] == "performance"
                            and r["layer"] == "model" and r["metric"] == metric and r["role"] in (role, "static")]
                    reference = {r["seed"]: r for r in pool if r["arm"] == "R4"}
                    if not reference:
                        raise ValueError("Missing R4 paired reference")
                    candidates = sorted({(r["arm"], r["variant"]) for r in pool if r["arm"] != "R4"})
                    for arm, variant in candidates:
                        comparison = {r["seed"]: r for r in pool if (r["arm"], r["variant"]) == (arm, variant)}
                        differences = []
                        for seed in config["suite"]["seeds"]:
                            a, b = reference[seed]["value"], comparison[None if None in comparison else seed]["value"]
                            differences.append(None if a is None or b is None else factor * (a - b))
                        result.append({"task": task, "k": k, "role": role, "reference": "R4", "comparison": arm,
                                       "variant": variant, "metric": metric, "unit": unit,
                                       "seed_differences": dict(zip(map(str, config["suite"]["seeds"]), differences)),
                                       **seed_summary(differences)})
    return result
