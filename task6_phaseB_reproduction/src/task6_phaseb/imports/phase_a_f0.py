from __future__ import annotations

from pathlib import Path

from task6_phaseb.common.config import analysis_id, protocol_id, root_for
from task6_phaseb.common.io import checked_complete, complete, fresh_output, read_json, sha256, write_json


ARM_MAP = {
    ("R2", "default"): ("R2", "default"),
    ("R4", "default"): ("R4o", "default"),
    ("R4-R2Init", "default"): ("R4d", "default"),
    ("G1", "default"): ("G1", "default"),
    ("G2", "aux_0.001"): ("G2-0.001", "default"),
    ("G4", "default"): ("G4", "default"),
}
METRIC_WHITELIST = {
    "performance": {"accuracy", "invalid_rate", "relative_performance", "count", "correct", "invalid"},
    "load_balance": {"cv", "valid_token_count", "cv_layer_std", "cv_worst"},
    "churn": {"churn", "exact_set_change", "valid_token_count", "churn_layer_std", "exact_set_change_layer_std", "churn_worst", "exact_set_change_worst"},
    "oracle_overlap": {"oracle_overlap", "valid_token_count", "oracle_overlap_layer_std", "oracle_overlap_worst"},
    "activation_coverage": {"activation_coverage", "valid_token_count", "zero_activation_count", "activation_coverage_layer_std", "activation_coverage_worst"},
}


def import_path(config, run_id):
    return root_for(config) / "artifacts/imports/phase_a_f0" / run_id


def _source(config):
    spec = config.get("imports", {}).get("phase_a_f0")
    if not spec:
        raise ValueError("Missing imports.phase_a_f0 configuration")
    path = Path(spec["normalized_metrics"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase A/F0 normalized result: {path}")
    expected = spec["expected_sha256"]
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("imports.phase_a_f0.expected_sha256 must be a lowercase SHA256")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Phase A/F0 source hash mismatch: {actual}")
    return spec, path, actual


def _convert_row(row):
    group, metric = row.get("group"), row.get("metric")
    if group not in METRIC_WHITELIST or metric not in METRIC_WHITELIST[group]:
        return None
    if row["arm"] == "dense":
        if group != "performance" or row.get("layer") != "model" or row.get("role") != "static":
            return None
        return {**row, "experts": 0, "k": 0, "source": "phase_a_f0_import"}
    mapped = ARM_MAP.get((row["arm"], row.get("variant", "default")))
    if mapped is None or row.get("k") not in (6, 13, 19, 26):
        return None
    arm, variant = mapped
    converted = {
        **row,
        "arm": arm,
        "variant": variant,
        "experts": 64,
        "source": "phase_a_f0_import",
    }
    return converted


def _add_adjusted_overlap(rows):
    additions = []
    for row in rows:
        if row["group"] == "oracle_overlap" and row["metric"] == "oracle_overlap" and row["value"] is not None:
            baseline = row["k"] / 64
            additions.append({
                **row,
                "group": "adjusted_overlap",
                "metric": "adjusted_overlap",
                "value": (row["value"] - baseline) / (1 - baseline),
            })
    return [*rows, *additions]


def import_phase_a_f0(config, run_id):
    spec, source, source_hash = _source(config)
    destination = import_path(config, run_id)
    header = {
        "schema": 1,
        "protocol": protocol_id(config),
        "analysis": analysis_id(config),
        "source": str(source),
        "source_sha256": source_hash,
        "source_run_id": spec["source_run_id"],
    }
    if destination.exists():
        checked_complete(destination, header)
        print(f"Verified existing Phase A/F0 import: {destination}")
        return
    payload = read_json(source)
    rows = [_convert_row(row) for row in payload["rows"]]
    rows = _add_adjusted_overlap([row for row in rows if row is not None])
    tasks = set(config["suite"]["tasks"])
    rows = [row for row in rows if row["task"] in tasks]
    dense = {
        row["task"]: row["value"] for row in rows
        if row["arm"] == "dense" and row["metric"] == "accuracy"
    }
    if set(dense) != tasks:
        raise ValueError(f"Imported Dense-init accuracy is incomplete: {dense}")
    observed = {
        (row["task"], row["experts"], row["k"], row["arm"])
        for row in rows if row["arm"] != "dense" and row["group"] == "performance"
        and row["metric"] == "accuracy" and row["layer"] == "model" and row["role"] in ("best", "static")
    }
    expected = {
        (task, 64, k, arm)
        for task in tasks for k in (6, 13, 19, 26) for arm in ("R2", "R4o", "R4d", "G1", "G2-0.001", "G4")
    }
    if observed != expected:
        raise ValueError(f"Imported E64 performance matrix mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    with fresh_output(destination):
        write_json(destination / "normalized.json", {"header": header, "dense_accuracy": dense, "rows": rows})
        complete(destination, header)
    print(f"Imported {len(rows)} E64/Dense normalized rows from {source}")


def load_import(config, run_id):
    _, source, source_hash = _source(config)
    spec = config["imports"]["phase_a_f0"]
    header = {
        "schema": 1,
        "protocol": protocol_id(config),
        "analysis": analysis_id(config),
        "source": str(source),
        "source_sha256": source_hash,
        "source_run_id": spec["source_run_id"],
    }
    directory = import_path(config, run_id)
    checked_complete(directory, header)
    payload = read_json(directory / "normalized.json")
    if payload["header"] != header:
        raise ValueError("Imported payload header mismatch")
    return payload


def baseline_accuracy(config, run_id, task):
    payload = load_import(config, run_id)
    return float(payload["dense_accuracy"][task])
