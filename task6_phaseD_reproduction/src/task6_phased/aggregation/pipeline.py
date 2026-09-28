from __future__ import annotations

from collections import defaultdict
import csv
import math
import shutil

import numpy as np

from task6_phased.common.config import (
    TASKS, analysis_id, budgets_for, conditions, dense_run_path, protocol_id,
    routed_root, run_path,
)
from task6_phased.common.io import read_json, sha256, write_json
from task6_phased.dense.model import resolve_asset
from task6_phased.metrics.performance.pipeline import best_state
from task6_phased.metrics.specialization.core import js_distance
from task6_phased.training.checkpoints import states


def _dense_reference(config):
    manifest = read_json(resolve_asset(config, "dense_best") / "phase_d_dense_manifest.json")
    run_id = manifest["source_run_id"]
    selection = read_json(dense_run_path(config, "evaluation", run_id) / "selection.json")
    data = read_json(dense_run_path(config, "evaluation", run_id) / f"{selection['state']['name']}.json")
    return manifest, data


def _flatten(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append((prefix, float(value)))
    return rows


def collect(config, run_id):
    dense_manifest, dense = _dense_reference(config)
    rows = []
    for condition in conditions(config):
        best = best_state(config, condition, run_id)["name"]
        final = f"step_{config['training']['total_optimizer_steps']}" if condition.trainable else "static"
        for state in states(config, condition, run_id):
            performance = read_json(
                run_path(config, "metrics/performance", condition, run_id) / state["name"] / "metrics.json"
            )
            for task in TASKS:
                metrics = performance["tasks"][task]
                for metric in ("native", "greedy_exact", "greedy_invalid_rate"):
                    rows.append({
                        **condition.to_dict(), "state": state["name"], "step": state["step"],
                        "group": "performance", "domain": task, "metric": metric,
                        "value": metrics[metric], "is_best": state["name"] == best,
                        "is_final": state["name"] == final,
                    })
                if state["name"] == best:
                    rows.append({
                        **condition.to_dict(), "state": state["name"], "step": state["step"],
                        "group": "performance", "domain": task, "metric": "delta_vs_dense_best",
                        "value": metrics["native"] - dense[task]["metrics"]["native"],
                        "is_best": True, "is_final": state["name"] == final,
                    })
            for metric in ("macro", "worst_domain"):
                rows.append({
                    **condition.to_dict(), "state": state["name"], "step": state["step"],
                    "group": "performance", "domain": "all", "metric": metric,
                    "value": performance["summary"][metric], "is_best": state["name"] == best,
                    "is_final": state["name"] == final,
                })
            for group in ("load_balance", "selection_quality", "specialization", "churn"):
                path = run_path(config, f"metrics/{group}", condition, run_id) / state["name"] / "metrics.json"
                if not path.is_file():
                    continue
                payload = read_json(path)
                for key, value in _flatten({k: v for k, v in payload.items() if k not in {
                    "protocol", "analysis", "condition", "state", "metric_group"
                }}):
                    rows.append({
                        **condition.to_dict(), "state": state["name"], "step": state["step"],
                        "group": group, "domain": "structured", "metric": key, "value": value,
                        "is_best": state["name"] == best, "is_final": state["name"] == final,
                    })
    return dense_manifest, dense, rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        roles = [("trajectory", row["state"], row["step"])]
        if row["is_best"]:
            roles.append(("best", "best", None))
        if row["is_final"]:
            roles.append(("final", "final", row["step"]))
        for role, state, step in roles:
            key = (
                row["experts"], row["arm"], row["variant"], row["k"], state, step,
                row["group"], row["domain"], row["metric"], role,
            )
            grouped[key].append(row)
    result = []
    for key, items in grouped.items():
        experts, arm, variant, k, state, step, group, domain, metric, role = key
        values = [item["value"] for item in items]
        deterministic = arm == "R2"
        # Whether a best checkpoint is also final varies by seed.  It is metadata,
        # not part of the statistical condition, so it must never split the seed
        # aggregate into multiple rows.  At aggregate level the flag means that
        # every contributing state has the other role as well.
        is_best = role == "best" or (role == "final" and all(item["is_best"] for item in items))
        is_final = role == "final" or (role == "best" and all(item["is_final"] for item in items))
        result.append({
            "experts": experts, "arm": arm, "variant": variant, "k": k,
            "state": state, "step": step,
            "group": group, "domain": domain, "metric": metric,
            "role": role, "is_best": is_best, "is_final": is_final, "n": len(values),
            "mean": float(np.mean(values)),
            "std": None if deterministic or len(values) < 2 else float(np.std(values, ddof=1)),
            "deterministic": deterministic,
        })
    return result


def paired_best(rows):
    performance = [
        row for row in rows
        if row["group"] == "performance" and row["is_best"]
        and row["metric"] in ("native", "macro", "worst_domain")
    ]
    lookup = {
        (row["experts"], row["arm"], row["k"], row["seed"], row["domain"], row["metric"]): row["value"]
        for row in performance if row["seed"] is not None
    }
    result = []
    for experts in sorted({row["experts"] for row in performance}):
        for arm in ("R4o", "R4d", "G2-0.001", "G4"):
            for k in budgets_for_from_rows(rows, experts):
                for domain in (*TASKS, "all"):
                    metric = "native" if domain != "all" else "macro"
                    values = [
                        lookup[(experts, arm, k, seed, domain, metric)]
                        - lookup[(experts, "G1", k, seed, domain, metric)]
                        for seed in (0, 1, 2)
                    ]
                    result.append({
                        "experts": experts, "comparison": arm, "reference": "G1",
                        "k": k, "domain": domain, "metric": metric,
                        "seed_differences": dict(zip(("0", "1", "2"), values)),
                        "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                        "checkpoint_role": "best",
                    })
    return result


def budgets_for_from_rows(rows, experts):
    return sorted({int(row["k"]) for row in rows if int(row["experts"]) == int(experts)})


def same_domain_seed_baseline(config, run_id):
    """JS between independently trained seeds for the same domain at each seed's best state."""
    by_key = {}
    for condition in conditions(config):
        if not condition.trainable:
            continue
        state = best_state(config, condition, run_id)
        payload = read_json(
            run_path(config, "metrics/specialization", condition, run_id) / state["name"] / "metrics.json"
        )
        layers = payload["populations"]["encoder_content"]["layers"]
        for layer, values in layers.items():
            for task in TASKS:
                counts = np.asarray(values["domain_counts"][task], dtype=np.float64)
                by_key[(condition.experts, condition.arm, condition.k, condition.seed, layer, task)] = counts / counts.sum()
    rows = []
    for experts in sorted({condition.experts for condition in conditions(config)}):
        for arm in ("R4o", "R4d", "G1", "G2-0.001", "G4"):
            for k in budgets_for(config, experts):
                for task in TASKS:
                    values = []
                    for left, right in ((0, 1), (0, 2), (1, 2)):
                        for layer in (
                            f"encoder_layer_{index:02d}" for index in range(config["model"]["encoder_layers"])
                        ):
                            value = js_distance(
                                by_key[(experts, arm, k, left, layer, task)],
                                by_key[(experts, arm, k, right, layer, task)],
                            )
                            values.append(value)
                            rows.append({
                                "experts": experts, "arm": arm, "k": k, "domain": task,
                                "population": "encoder_content", "layer": layer,
                                "seed_pair": f"{left}__{right}", "js_distance": value,
                                "checkpoint_role": "best_per_seed",
                            })
                    rows.append({
                        "experts": experts, "arm": arm, "k": k, "domain": task,
                        "population": "encoder_content", "layer": "all_encoder_mean",
                        "seed_pair": "all_three_pairs", "js_distance": float(np.mean(values)),
                        "checkpoint_role": "best_per_seed",
                    })
    return rows


def _existing_e64_rows(root, name):
    """Import the completed routed02 E64 result without reopening old checkpoints.

    The original E64 protocol identity includes the old implementation hash.  Its
    normalized result is therefore the immutable compatibility boundary for this
    supplement.  Older aggregate rows omitted the constant experts column, so it
    is restored explicitly here.
    """
    path = root / name / "metrics.json"
    if not path.is_file():
        if name in ("normalized", "seed_baseline"):
            raise FileNotFoundError(
                f"The E128/E256 supplement requires existing E64 {name}: {path}"
            )
        return []
    result = []
    for row in read_json(path)["rows"]:
        experts = int(row.get("experts", 64))
        if experts == 64:
            result.append({**row, "experts": 64})
    if not result and name in ("normalized", "seed_baseline"):
        raise ValueError(f"Existing result contains no E64 rows: {path}")
    return result


def _legacy_e64_snapshot(root):
    """Preserve the pre-supplement E64 aggregate before writing combined data."""
    target = root / "legacy_e64_source"
    if target.is_dir():
        identity = read_json(target / "source.json")
        for name, digest in identity["files"].items():
            if sha256(target / name) != digest:
                raise ValueError(f"Corrupt E64 source snapshot: {name}")
        return target
    for name in ("normalized", "seed_baseline"):
        for filename in ("metrics.json", "metrics.csv"):
            source = root / name / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing completed E64 source result: {source}")
            destination = target / name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    write_json(target / "source.json", {
        "role": "immutable pre-supplement routed02 E64 aggregate",
        "files": {
            f"{name}/{filename}": sha256(target / name / filename)
            for name in ("normalized", "seed_baseline")
            for filename in ("metrics.json", "metrics.csv")
        },
    })
    return target


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(config, run_id):
    root = routed_root(config) / "results" / run_id / "data"
    legacy_root = _legacy_e64_snapshot(root)
    legacy_normalized = _existing_e64_rows(legacy_root, "normalized")
    legacy_seed_baseline = _existing_e64_rows(legacy_root, "seed_baseline")
    dense_manifest, dense, supplemental = collect(config, run_id)
    normalized = [*legacy_normalized, *supplemental]
    aggregated = summarize(normalized)
    paired = paired_best(normalized)
    seed_baseline = [*legacy_seed_baseline, *same_domain_seed_baseline(config, run_id)]
    meta = {
        "protocol": protocol_id(config), "analysis": analysis_id(config), "run_id": run_id,
        "dense_manifest": dense_manifest, "performance_difference_checkpoint_role": "best_only",
        "legacy_e64_import": read_json(legacy_root / "source.json"),
    }
    for name, rows in (
        ("normalized", normalized), ("aggregated", aggregated),
        ("paired_differences", paired), ("seed_baseline", seed_baseline),
    ):
        write_json(root / name / "metrics.json", {"meta": meta, "rows": rows})
        _write_csv(root / name / "metrics.csv", rows)
    write_json(root / "dense_reference.json", {"manifest": dense_manifest, "evaluation": dense})
    return root
