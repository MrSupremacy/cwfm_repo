from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from task8_p01.assets.catalog import read_catalog
from task8_p01.common.config import BUDGETS, EXPERTS, SEEDS, TASKS, resolve_path
from task8_p01.common.io import checked_complete, read_json, write_csv
from task8_p01.compat.task6 import load_task6_config


def _output_root(config):
    return resolve_path(config, config["execution"]["output_root"])


def _read_result(path):
    checked_complete(path)
    identity = read_json(path / "identity.json")
    summary = read_json(path / "summary.json")
    tasks = {}
    for task in TASKS:
        with gzip.open(path / f"{task}.json.gz", "rt", encoding="utf-8") as stream:
            tasks[task] = json.load(stream)["metrics"]
    return identity, summary, tasks


def collect_full_results(config):
    root = _output_root(config) / "runs/evaluation/full_validation"
    rows = []
    if not root.exists():
        return rows
    for identity_file in sorted(root.rglob("identity.json")):
        path = identity_file.parent
        identity, summary, tasks = _read_result(path)
        shared = {
            "phase": "P1", "E": identity["experts"], "k": identity["k"],
            "seed": identity["seed"], "checkpoint_role": identity["checkpoint_role"],
            "checkpoint_step": identity["checkpoint_step"],
            "checkpoint_hash": identity["checkpoint_hash"], "mode": identity["mode"],
            "selector_id": "R2_C0_L2" if identity["mode"] in ("M00", "M01") else "R4d_St_RMS",
            "weighting_id": "uniform" if identity["mode"] in ("M00", "M10") else "R4d_St_RMS_ksoftmax",
            "aggregation_id": "masked_neuron_sum", "dense_hash": identity["dense_hash"],
            "task6_source_hash": identity["task6_source_hash"], "status": "complete",
            "artifact_path": str(path),
        }
        for task in TASKS:
            rows.append({**shared, "task": task, **tasks[task]})
        rows.append({**shared, "task": "__macro__", "native": summary["macro"], "count": ""})
        rows.append({**shared, "task": "__worst__", "native": summary["worst_domain"], "count": ""})
    return rows


def factorial_effects(rows):
    lookup = {}
    for row in rows:
        key = (
            int(row["E"]), int(row["k"]), row["checkpoint_role"],
            row["seed"], row["task"], row["mode"],
        )
        lookup[key] = float(row["native"])
    result = []
    learned_snapshots = sorted({
        (int(row["E"]), int(row["k"]), row["checkpoint_role"], row["seed"])
        for row in rows if row["mode"] != "M00"
    }, key=lambda item: (item[0], item[1], item[2], -1 if item[3] is None else item[3]))
    for experts, k, role, seed in learned_snapshots:
        for task in (*TASKS, "__macro__", "__worst__"):
            values = {}
            for mode in ("M00", "M01", "M10", "M11"):
                key = (
                    experts, k, "static" if mode == "M00" else role,
                    None if mode == "M00" else seed, task, mode,
                )
                if key not in lookup:
                    break
                values[mode] = lookup[key]
            if len(values) != 4:
                continue
            m00, m01, m10, m11 = (values[mode] for mode in ("M00", "M01", "M10", "M11"))
            result.append({
                "E": experts, "k": k, "seed": seed, "checkpoint_role": role, "task": task,
                **values,
                "aggregation_R2": m01 - m00,
                "aggregation_R4": m11 - m10,
                "selection_uniform": m10 - m00,
                "selection_soft": m11 - m01,
                "interaction": m11 - m10 - m01 + m00,
                "selector_marginal_A": ((m10 - m00) + (m11 - m01)) / 2,
                "aggregation_marginal_B": ((m01 - m00) + (m11 - m10)) / 2,
            })
    return result


def _old_summary(config, experts, k, seed, arm, state_name):
    task6_config = load_task6_config(config)
    from task6_phased.common.config import Condition, run_path

    condition = Condition(experts, arm, k, seed)
    path = run_path(task6_config, "capture/A", condition, config["protocol"]["source_run_id"]) / state_name
    if not path.exists():
        return None, None
    value = read_json(path / "summary.json")
    return float(value["macro"]), str(path)


def endpoint_reproduction(config, rows):
    macro = {
        (int(row["E"]), int(row["k"]), row["checkpoint_role"], row["seed"], row["mode"]): row
        for row in rows if row["task"] == "__macro__"
    }
    catalog = read_catalog(config)
    result = []
    for experts in EXPERTS:
        for k in BUDGETS[experts]:
            new = macro.get((experts, k, "static", None, "M00"))
            old, old_path = _old_summary(config, experts, k, None, "R2", "static")
            result.append({
                "E": experts, "k": k, "arm": "R2", "seed": "", "checkpoint_role": "static",
                "step": 0, "old_macro": old, "new_macro": None if new is None else new["native"],
                "abs_diff": None if old is None or new is None else abs(old - float(new["native"])),
                "checkpoint_hash": "", "status": "missing" if old is None or new is None else "complete",
                "old_artifact": old_path, "new_artifact": None if new is None else new["artifact_path"],
            })
            init_entry = catalog["entries"][f"E{experts}_k{k}_init"]
            new = macro.get((experts, k, "init", None, "M11"))
            init_seed = init_entry["canonical_source_seed"]
            old, old_path = _old_summary(config, experts, k, init_seed, "R4d", "step_0")
            result.append({
                "E": experts, "k": k, "arm": "R4d", "seed": "", "checkpoint_role": "init",
                "step": 0, "old_macro": old, "new_macro": None if new is None else new["native"],
                "abs_diff": None if old is None or new is None else abs(old - float(new["native"])),
                "checkpoint_hash": init_entry["summary_sha256"],
                "status": "missing" if old is None or new is None else "complete",
                "old_artifact": old_path, "new_artifact": None if new is None else new["artifact_path"],
            })
            for seed in SEEDS:
                entry = catalog["entries"][f"E{experts}_k{k}_seed{seed}_best"]
                new = macro.get((experts, k, "best", seed, "M11"))
                old = entry.get("old_macro")
                if old is None:
                    old, old_path = _old_summary(config, experts, k, seed, "R4d", entry["state_name"])
                else:
                    old_path = entry["selection_path"]
                result.append({
                    "E": experts, "k": k, "arm": "R4d", "seed": seed, "checkpoint_role": "best",
                    "step": entry["step"], "old_macro": old,
                    "new_macro": None if new is None else new["native"],
                    "abs_diff": None if old is None or new is None else abs(float(old) - float(new["native"])),
                    "checkpoint_hash": entry["state_sha256"],
                    "status": "missing" if old is None or new is None else "complete",
                    "old_artifact": old_path, "new_artifact": None if new is None else new["artifact_path"],
                })
    return result


def aggregate_results(config, run_id="p01"):
    rows = collect_full_results(config)
    effects = factorial_effects(rows)
    endpoints = endpoint_reproduction(config, rows)
    root = _output_root(config) / "results" / run_id / "tables"
    replay = []
    replay_root = _output_root(config) / "runs/local_replay"
    if replay_root.exists():
        for path in sorted(replay_root.rglob("p1_local_replay.csv")):
            checked_complete(path.parent)
            with path.open(encoding="utf-8", newline="") as stream:
                replay.extend(csv.DictReader(stream))
    write_csv(root / "p1_full_four_cell.csv", rows)
    write_csv(root / "p1_effects.csv", effects)
    write_csv(root / "p1_local_replay.csv", replay)
    write_csv(root / "p0_endpoint_reproduction.csv", endpoints)
    return {
        "full_rows": len(rows), "effect_rows": len(effects), "local_replay_rows": len(replay),
        "endpoint_rows": len(endpoints), "root": str(root),
    }
