from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from task6.aggregation.core import seed_summary
from task6.common.config import analysis_id, protocol_id, root_for
from task6.common.io import read_json, sha256, write_json

ARM_FROM_F0 = {
    "R2": "R2", "R2-soft": "R2-soft", "R4": "R4o",
    "R4-R2Init": "R4d", "R4-hard": "R4o-hard",
    "G1": "G1", "G2": "G2", "G4": "G4",
}
GROUPS = {"performance", "load_balance", "churn", "oracle_overlap", "activation_coverage"}


def compare(config, run_id, source):
    source = Path(source).resolve()
    old = read_json(source)
    f1_path = root_for(config) / "results/data/normalized" / run_id / "metrics.json"
    current = read_json(f1_path)
    if (current["meta"]["protocol"], current["meta"]["analysis"]) != (
            protocol_id(config), analysis_id(config)):
        raise ValueError("F1 normalized results belong to a different implementation/configuration")

    old_rows = []
    for row in old["rows"]:
        mapped = ARM_FROM_F0.get(row.get("arm"))
        if mapped and row.get("group") in GROUPS:
            old_rows.append({**row, "mapped_arm": mapped})
    present = {row["mapped_arm"] for row in old_rows}
    if present != set(ARM_FROM_F0.values()):
        raise ValueError(f"Task 5 F0 source has the wrong shared arm set: {sorted(present)}")

    index = defaultdict(list)
    for row in old_rows:
        epoch = row.get("epoch") if row["role"] == "trajectory" else None
        key = (row["task"], row["mapped_arm"], row["k"], row["role"], row["group"],
               row["layer"], row["metric"], epoch)
        index[key].append(row)

    normalized = []
    for row in current["rows"]:
        if row["arm"] not in set(ARM_FROM_F0.values()) or row["group"] not in GROUPS:
            continue
        f0_role = "static" if row["arm"] in {"R2", "R2-soft"} and row["role"] in {"best", "final"} else row["role"]
        if f0_role == "static":
            epoch = None
        elif row["role"] == "trajectory":
            epoch = row["epoch"]
        else:
            epoch = None
        key = (row["task"], row["arm"], row["k"], f0_role, row["group"],
               row["layer"], row["metric"], epoch)
        candidates = index.get(key, [])
        if not candidates:
            continue  # F0 legitimately lacks R2/R2-soft trajectories and some historical D trajectories.
        matches = [item for item in candidates if item.get("seed") in (None, row["seed"])]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or missing F0 comparison row: {key}, seed={row['seed']}")
        f0 = matches[0]
        delta = None if row["value"] is None or f0["value"] is None else row["value"] - f0["value"]
        normalized.append({
            "task": row["task"], "arm": row["arm"], "variant": row["variant"],
            "k": row["k"], "seed": row["seed"], "role": row["role"],
            "epoch": row.get("epoch"), "group": row["group"], "layer": row["layer"],
            "metric": row["metric"], "f0_role": f0_role,
            "f0_value": f0["value"], "f1_value": row["value"], "delta_f1_minus_f0": delta,
        })

    grouped = defaultdict(list)
    identity_fields = ("task", "arm", "variant", "k", "role", "epoch", "group", "layer", "metric", "f0_role")
    for row in normalized:
        grouped[tuple(row[name] for name in identity_fields)].append(row)
    aggregated = []
    expected_seeds = set(config["suite"]["seeds"])
    for key, rows in grouped.items():
        if {row["seed"] for row in rows} != expected_seeds or len(rows) != len(expected_seeds):
            raise ValueError(f"Incomplete F0/F1 seed group: {key}")
        aggregated.append({
            **dict(zip(identity_fields, key)),
            "f0": seed_summary([row["f0_value"] for row in rows], key[-1] == "static"),
            "f1": seed_summary([row["f1_value"] for row in rows]),
            "delta_f1_minus_f0": seed_summary([row["delta_f1_minus_f0"] for row in rows]),
        })

    meta = {
        "protocol": protocol_id(config), "analysis": analysis_id(config), "run_id": run_id,
        "f0_source": str(source), "f0_sha256": sha256(source),
        "f0_meta": old.get("meta", {}),
        "interpretation": "F0 and F1 are separate regimes; same-numbered seeds are not paired random trials.",
    }
    destination = root_for(config) / "results/data/f0_f1_comparison" / run_id / "metrics.json"
    write_json(destination, {"meta": meta, "normalized": normalized, "aggregated": aggregated})
    print(f"Compared {len(normalized)} F0/F1 rows: {destination}")
