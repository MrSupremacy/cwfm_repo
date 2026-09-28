"""Phase A/F1 result completeness checks."""
from __future__ import annotations

from collections import defaultdict

from task6.aggregation.pipeline import aggregate_rows, collect_rows
from task6.common.config import ARMS, DENSE_FT_ARM, conditions
from task6.metrics.pipeline import layer_names


def validate_rows(rows, config):
    pool = defaultdict(list)
    for row in rows:
        pool[(row["task"], row["arm"], row["variant"], row["k"], row["seed"])].append(row)
    expected_conditions = [condition for condition in conditions(config) if condition.trainable]
    expected = {(c.task, c.arm, c.variant, c.k, c.seed) for c in expected_conditions}
    actual = {key for key in pool if key[1] != "dense"}
    if actual != expected:
        raise ValueError("Phase A result rows do not cover the exact 198-run matrix")
    epochs = set(range(config["training"]["epochs"] + 1))
    sparse_conditions = [condition for condition in expected_conditions if condition.is_routed]
    for condition in sparse_conditions:
        items = pool[(condition.task, condition.arm, condition.variant, condition.k, condition.seed)]
        for group, metric in (("performance", "accuracy"), ("load_balance", "cv"),
                              ("oracle_overlap", "oracle_overlap"),
                              ("activation_coverage", "activation_coverage")):
            observations = [r for r in items if r["group"] == group and r["metric"] == metric]
            roles = {r["role"] for r in observations if r["layer"] == "model"}
            if not {"best", "final"} <= roles:
                raise ValueError(f"Missing best/final {group}: {condition}")
        for group, metric in (("performance", "accuracy"), ("oracle_overlap", "oracle_overlap"),
                              ("activation_coverage", "activation_coverage")):
            for layer in (["model"] if group == "performance" else ["model", *layer_names(config)]):
                seen = {r["epoch"] for r in items if r["group"] == group and r["metric"] == metric
                        and r["layer"] == layer and r["role"] == "trajectory"}
                if seen != epochs:
                    raise ValueError(f"Incomplete trajectory {group}/{layer}: {condition}")
        for layer in ["model", *layer_names(config)]:
            seen = {r["epoch"] for r in items if r["group"] == "churn" and r["metric"] == "churn"
                    and r["layer"] == layer and r["role"] == "trajectory"}
            if seen != epochs - {0}:
                raise ValueError(f"Incomplete churn trajectory {layer}: {condition}")
    if {c.arm for c in sparse_conditions} != set(ARMS):
        raise ValueError("The Phase A/F1 arm set changed")
    dense_ft = [condition for condition in expected_conditions if condition.arm == DENSE_FT_ARM]
    expected_dense_ft = len(config["suite"]["tasks"]) * len(config["suite"]["seeds"])
    if len(dense_ft) != expected_dense_ft:
        raise ValueError("Dense-fullFT must contain exactly task × seed runs without k expansion")
    for condition in dense_ft:
        items = pool[(condition.task, condition.arm, condition.variant, condition.k, condition.seed)]
        accuracy = [row for row in items if row["group"] == "performance"
                    and row["metric"] == "accuracy" and row["layer"] == "model"]
        if not {"best", "final"} <= {row["role"] for row in accuracy}:
            raise ValueError(f"Missing Dense-fullFT best/final performance: {condition}")
        if {row["epoch"] for row in accuracy if row["role"] == "trajectory"} != epochs:
            raise ValueError(f"Incomplete Dense-fullFT performance trajectory: {condition}")
    aggregate_rows(rows, config)


def check(config, run_id):
    rows = collect_rows(config, run_id)
    validate_rows(rows, config)
    print("Phase A/F1 sparse matrix and Dense-fullFT baselines are complete and internally consistent")


def report(config, run_id):
    from task6.aggregation.pipeline import aggregate
    from task6.visualization.render import figures, tables

    aggregate(config, run_id)
    check(config, run_id)
    tables(config, run_id)
    figures(config, run_id)
    print("Phase A/F1 aggregate, tables, and figures completed")
