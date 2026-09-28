"""Read-only reuse of v3 F0 results plus two new arms; never writes old results."""
from __future__ import annotations

from collections import defaultdict

from task5.aggregation.pipeline import _meta, aggregate_rows, collect_rows, paired_differences
from task5.common.config import conditions, extension_spec, root_for, run_path
from task5.common.context import _shared_config_identity, shared_path, verify_prepared
from task5.common.io import checked_complete, read_json, sha256, write_json

NEW_ARMS = frozenset(("R2-soft", "R4-hard"))
BASE_METHODS = frozenset((arm, "default") for arm in ("R2", "R4", "R4-R2Init", "G1", "G4")) | {("G2", "aux_0.001")}
METHODS = BASE_METHODS | {(arm, "default") for arm in NEW_ARMS}


def result_root(config):
    # Fixed sibling of results; no CLI path can alias the existing result tree.
    root = root_for(config) / "task6_phaseA_F0_result"
    if root.resolve() == (root_for(config) / "results").resolve() or root.is_symlink():
        raise ValueError("Phase A F0 result destination must not alias existing results")
    return root


def require_phase_a(config, run_id):
    extension = extension_spec(config, run_id)
    if not extension or extension.get("arms") != ["R2-soft", "R4-hard"]:
        raise ValueError("Use configs/extensions/phaseA_f0.yaml for the F0 supplement")
    if {(v["arm"], v["name"]) for v in config["variants"]} != METHODS:
        raise ValueError("Phase A F0 comparison must contain exactly the eight specified methods")
    return extension


def validate_rows(rows, config, methods):
    """Reject missing conditions, seeds, layers and metric timelines, not just duplicates."""
    from task5.metrics.pipeline import layer_names
    pool = defaultdict(list)
    for row in rows:
        pool[(row["task"], row["arm"], row["variant"], row["k"], row["seed"])].append(row)
    selected = [c for c in conditions(config) if (c.arm, c.variant) in methods]
    expected = {(c.task, c.arm, c.variant, c.k, c.seed) for c in selected}
    if set(pool) != expected:
        raise ValueError("Comparison is missing/adding task/arm/k/seed conditions")
    all_epochs = set(range(config["training"]["epochs"] + 1))
    for c in selected:
        items = pool[(c.task, c.arm, c.variant, c.k, c.seed)]
        required = {("performance", "accuracy"): ["model"],
                    ("performance", "relative_performance"): ["model"],
                    ("load_balance", "cv"): ["model", *layer_names(config)],
                    ("oracle_overlap", "oracle_overlap"): ["model", *layer_names(config)],
                    ("activation_coverage", "activation_coverage"): ["model", *layer_names(config)],
                    ("coactivation_consistency", "ratio"): ["model", *layer_names(config)]}
        for (group, metric), layers in required.items():
            for layer in layers:
                observations = [r for r in items if (r["group"], r["metric"], r["layer"]) == (group, metric, layer)]
                roles = {r["role"] for r in observations}
                if not set(("best", "final") if c.trainable else ("static",)) <= roles:
                    raise ValueError(f"Missing selected/final metric: {c} {group} {layer}")
                if c.trainable and (group in ("performance", "coactivation_consistency") or
                                    (c.arm in ("R4", "R4-R2Init", "R4-hard") and group in ("oracle_overlap", "activation_coverage"))):
                    if {r["epoch"] for r in observations if r["role"] == "trajectory"} != all_epochs:
                        raise ValueError(f"Missing trajectory: {c} {group} {layer}")
        if c.trainable:
            for layer in ("model", *layer_names(config)):
                if {r["epoch"] for r in items if (r["group"], r["metric"], r["layer"], r["role"]) ==
                        ("churn", "churn", layer, "trajectory")} != all_epochs - {0}:
                    raise ValueError(f"Missing churn trajectory: {c} {layer}")
    aggregate_rows(rows, config)  # Also rejects duplicate rows / incomplete seed groups.


def compatible_config(recorded, current):
    if _shared_config_identity(recorded) != _shared_config_identity(current) or recorded["metrics"] != current["metrics"]:
        raise ValueError("F0 training/data/capture/metric settings differ from formal20260830a")


def check_base(config, run_id):
    """Read-only gate before expensive work; exact source and config provenance."""
    extension = require_phase_a(config, run_id)
    source = root_for(config) / "results/data/normalized" / run_id / "metrics.json"
    pinned = extension["source_report"]
    if sha256(source) != pinned["sha256"]:
        raise ValueError("Completed v3 normalized source changed; review it before changing the pinned hash")
    base = read_json(source)
    meta = base["meta"]
    if (meta["protocol"], meta["analysis"], meta["run_id"], meta["suite"]) != (
            pinned["protocol"], pinned["analysis"], run_id, config["suite"]["name"]):
        raise ValueError("Completed source report identity differs")
    if meta.get("extension_base", {}).get("protocol") != extension["base_protocol"]:
        raise ValueError("v3 report does not descend from the declared prepared base")
    rows = [r for r in base["rows"] if (r["arm"], r["variant"]) in BASE_METHODS]
    validate_rows(rows, config, BASE_METHODS)
    for task in config["suite"]["tasks"]:
        header = verify_prepared(config, task, run_id)
        original = read_json(shared_path(config, "probe_sets", task, run_id) / "context.json")["config"]
        compatible_config(original, config)
        e_header = checked_complete(shared_path(config, "coactivation", task, run_id))
        if e_header != {**header, "kind": "E", "neuron_order": "original", "population": "full_validation"}:
            raise ValueError("Shared dense coactivation reference differs from prepared base")
        # Dense A is a read-only denominator. Check file integrity and input identity.
        from task5.common.config import Condition
        from task5.metrics.performance.pipeline import captured_states
        from task5.metrics.pipeline import check_source
        dense = Condition(task, "dense")
        check_source(config, dense, run_id, captured_states(config, dense, run_id)[0], "A")
        for c in conditions(config):
            if c.task != task or not c.trainable or (c.arm, c.variant) not in BASE_METHODS:
                continue
            record = read_json(run_path(config, "train", c, run_id) / "config.json")
            if record["inputs"] != header or record["condition"] != c.to_dict():
                raise ValueError(f"Reused training input/condition mismatch: {c}")
            compatible_config(record["config"], config)
            old_variant = [v for v in record["config"]["variants"] if (v["arm"], v["name"]) == (c.arm, c.variant)]
            new_variant = [v for v in config["variants"] if (v["arm"], v["name"]) == (c.arm, c.variant)]
            if old_variant != new_variant:
                raise ValueError(f"Reused variant settings differ: {c}")
    result_root(config)
    print("Phase A F0 base verified: frozen backbone, same seeds/config/metrics, immutable v3 report + dense A/E")
    return rows, {**pinned, "path": str(source), "prepared_protocol": extension["base_protocol"]}


def report(config, run_id):
    from task5.visualization.render import figures, tables
    base_rows, provenance = check_base(config, run_id)
    selected = [c for c in conditions(config) if c.arm in NEW_ARMS]
    added = collect_rows(config, run_id, selected)
    validate_rows(added, config, {(arm, "default") for arm in NEW_ARMS})
    rows = [*base_rows, *added]
    validate_rows(rows, config, METHODS)
    meta = {**_meta(config, run_id, provenance), "phase": "A", "regime": "F0_router_only",
            "methods": sorted(METHODS), "coactivation_reference": "shared_original_dense",
            "note": "R2/R2-soft are static (one observation, no seed std/churn). R4-hard uses coefficient ST. "
                    "G1/G2/G4 retain only best/final local-q; missing probe trajectories are not imputed."}
    root = result_root(config)
    for section, name, values in (("normalized", "metrics.json", rows),
                                  ("aggregated", "metrics.json", aggregate_rows(rows, config)),
                                  ("aggregated", "paired_differences.json", paired_differences(rows, config))):
        write_json(root / "data" / section / run_id / name, {"meta": meta, "rows": values})
    tables(config, run_id, result_root=root)
    figures(config, run_id, result_root=root, random_reference_arm="R2")
    print(f"Phase A F0 eight-method tables/figures: {root}; existing results unchanged")
