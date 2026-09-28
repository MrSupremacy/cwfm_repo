from __future__ import annotations

from pathlib import Path
import csv
import json

from task6_phaseb.common.config import ARMS, conditions, protocol_id, root_for, run_path
from task6_phaseb.common.io import read_json


COLORS = {
    "R2": "#7f7f7f", "R4o": "#d62728", "R4d": "#e377c2",
    "G1": "#1f77b4", "G2-0.001": "#2ca02c", "G4": "#17becf",
}
MARKERS = {"R2": "o", "R4o": "s", "R4d": "D", "G1": "^", "G2-0.001": "v", "G4": "P"}
BUDGETS = {
    "ratio10": {64: 6, 128: 13, 256: 26},
    "ratio20": {64: 13, 128: 26, 256: 51},
    "ratio30": {64: 19, 128: 38, 256: 77},
    "ratio40": {64: 26, 128: 51, 256: 102},
}
PROPORTIONS = {
    "accuracy", "churn", "exact_set_change", "oracle_overlap", "adjusted_overlap",
    "activation_coverage", "boundary_tie_rate", "all_zero_rate",
}


def _results_root(config, result_root=None):
    return Path(result_root) if result_root is not None else root_for(config) / "results"


def data_file(config, run_id, section, *, result_root=None):
    result = read_json(_results_root(config, result_root) / "data" / section / run_id / "metrics.json")
    if result["meta"]["protocol"] != protocol_id(config):
        raise ValueError(f"Result protocol mismatch: {section}")
    return result


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot render empty required table: {path}")
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _main_role(row):
    return row["role"] == ("static" if row["arm"] in ("dense", "R2") else "best")


def _complete_count(path, pattern):
    return sum(1 for _ in Path(path).glob(pattern)) if Path(path).exists() else 0


def _probe_with_q_count(path):
    if not Path(path).exists():
        return 0
    return sum(
        bool(read_json(marker)["header"]["with_q"])
        for marker in Path(path).glob("*/probe/complete.json")
    )


def _artifact_rows(config, run_id):
    rows = []
    for condition in conditions(config):
        train = run_path(config, "train", condition, run_id)
        validation = run_path(config, "capture/validation", condition, run_id)
        probe = run_path(config, "capture/probe", condition, run_id)
        expected_states = config["training"]["epochs"] + 1 if condition.trainable else 1
        expected_d_min = expected_states if condition.arm in ("R4o", "R4d") else 1
        expected_d_max = expected_states if condition.arm in ("R4o", "R4d") else (2 if condition.trainable else 1)
        observed_checkpoints = _complete_count(train / "checkpoints", "*/complete.json") if condition.trainable else 0
        observed_a = _complete_count(validation, "*/A/complete.json")
        observed_b = _complete_count(validation, "*/B/complete.json")
        observed_probe = _complete_count(probe, "*/probe/complete.json")
        observed_d = _probe_with_q_count(probe)
        rows.append({
            **condition.to_dict(),
            "expected_checkpoints": expected_states if condition.trainable else 0,
            "observed_checkpoints": observed_checkpoints,
            "expected_A": expected_states,
            "observed_A": observed_a,
            "expected_B_max": 2 if condition.trainable else 1,
            "expected_B_min": 1,
            "observed_B": observed_b,
            "expected_C": expected_states,
            "observed_probe": observed_probe,
            "expected_D_min": expected_d_min,
            "expected_D_max": expected_d_max,
            "observed_D": observed_d,
            "complete": (
                observed_checkpoints == (expected_states if condition.trainable else 0)
                and observed_a == expected_states
                and 1 <= observed_b <= (2 if condition.trainable else 1)
                and observed_probe == expected_states
                and expected_d_min <= observed_d <= expected_d_max
            ),
        })
    return rows


def _resource_rows(config, run_id):
    result = []
    layers, width = config["model"]["encoder_layers"] + config["model"]["decoder_layers"], config["model"]["d_model"]
    artifact = _artifact_rows(config, run_id)
    for task in config["suite"]["tasks"]:
        for k in config["experiment"]["budgets"][64]:
            for arm in ARMS:
                trained = arm != "R2"
                params = 0 if not trained else layers * 64 * (width if arm in ("R4o", "R4d") else width + 1)
                result.append({
                    "source": "phase_a_f0_import", "task": task, "experts": 64, "k": k,
                    "budget_ratio": k / 64, "arm": arm, "router_parameters_per_run": params,
                    "condition_count": 3 if trained else 1,
                    "expected_checkpoints": 33 if trained else 0, "observed_checkpoints": "imported_summary_only",
                    "expected_A": 33 if trained else 1, "observed_A": "imported_summary_only",
                    "expected_B_min": 3 if trained else 1, "expected_B_max": 6 if trained else 1,
                    "observed_B": "imported_summary_only", "expected_C": 33 if trained else 1,
                    "observed_C": "imported_summary_only",
                    "expected_D_min": 33 if arm in ("R4o", "R4d") else (3 if trained else 1),
                    "expected_D_max": 33 if arm in ("R4o", "R4d") else (6 if trained else 1),
                    "observed_D": "imported_summary_only",
                })
        for experts in (128, 256):
            for k in config["experiment"]["budgets"][experts]:
                for arm in ARMS:
                    items = [r for r in artifact if (r["task"], r["experts"], r["k"], r["arm"]) == (task, experts, k, arm)]
                    params = 0 if arm == "R2" else layers * experts * (width if arm in ("R4o", "R4d") else width + 1)
                    result.append({
                        "source": "phase_b_new", "task": task, "experts": experts, "k": k, "budget_ratio": k / experts,
                        "arm": arm, "router_parameters_per_run": params,
                        "condition_count": len(items),
                        "expected_checkpoints": sum(r["expected_checkpoints"] for r in items),
                        "observed_checkpoints": sum(r["observed_checkpoints"] for r in items),
                        "expected_A": sum(r["expected_A"] for r in items),
                        "observed_A": sum(r["observed_A"] for r in items),
                        "expected_B_max": sum(r["expected_B_max"] for r in items),
                        "expected_B_min": sum(r["expected_B_min"] for r in items),
                        "observed_B": sum(r["observed_B"] for r in items),
                        "expected_C": sum(r["expected_C"] for r in items),
                        "observed_C": sum(r["observed_probe"] for r in items),
                        "expected_D_min": sum(r["expected_D_min"] for r in items),
                        "expected_D_max": sum(r["expected_D_max"] for r in items),
                        "observed_D": sum(r["observed_D"] for r in items),
                    })
    return result



