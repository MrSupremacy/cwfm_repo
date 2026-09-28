from __future__ import annotations

import argparse
import json

from task6_phaseb.common.config import ARMS, conditions, expected_matrix_counts, load_config, validate_run_id


COMMANDS = (
    "matrix", "preflight", "import-e64", "prepare", "validate", "train",
    "capture", "metrics", "aggregate", "tables", "figures", "smoke",
)
METRICS = (
    "all", "performance", "load_balance", "churn", "oracle_overlap",
    "adjusted_overlap", "activation_coverage", "oracle_resolution",
)


def parser():
    value = argparse.ArgumentParser(
        description="Task 6 Phase B/F0: E128/E256 router-only training and E64-combined reporting"
    )
    value.add_argument("command", choices=COMMANDS)
    value.add_argument("--suite", help="Suite YAML; defaults to configs/suites/phase_b_f0.yaml")
    value.add_argument("--local", help="Machine-specific paths and execution device YAML")
    value.add_argument("--run-id", default="main01")
    value.add_argument("--task", choices=["sst2", "mnli"])
    value.add_argument("--experts", type=int, choices=[128, 256])
    value.add_argument("--arm", choices=list(ARMS))
    value.add_argument("--variant")
    value.add_argument("--k", type=int)
    value.add_argument("--seed", type=int)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--shard-count", type=int, default=1, help="Condition sharding; every run stays world-size 1")
    value.add_argument("--part", choices=["A", "select-best", "diagnostics", "all"], default="all")
    value.add_argument("--metric", choices=METRICS, default="all")
    value.add_argument("--resume", help="Checkpoint name in the one selected training run")
    value.add_argument("--stop-after-epoch", type=int)
    value.add_argument("--skip-complete", action="store_true")
    value.add_argument("--config-only", action="store_true")
    value.add_argument("--list", action="store_true")
    return value


def _has_filter(args):
    return any(getattr(args, key) is not None for key in ("task", "experts", "arm", "variant", "k", "seed"))


def select_conditions(config, args):
    selected = conditions(config)
    for key in ("task", "experts", "arm", "variant", "k", "seed"):
        wanted = getattr(args, key)
        if wanted is not None:
            selected = [condition for condition in selected if getattr(condition, key) == wanted]
    if args.command == "train":
        selected = [condition for condition in selected if condition.trainable]
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    selected = selected[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError("No matching Phase B conditions")
    return selected


def _complete_suite_only(args):
    if args.shard_count != 1 or _has_filter(args):
        raise ValueError("This stage requires the complete suite without condition filters")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run_id(args.run_id)
    config = load_config(args.suite, args.local)
    selected = select_conditions(config, args)

    if args.command in ("aggregate", "tables", "figures", "smoke", "import-e64"):
        _complete_suite_only(args)

    needs_torch = args.command in {"prepare", "validate", "train", "smoke"} or (
        args.command == "capture" and args.part != "select-best"
    )
    if needs_torch:
        from task6_phaseb.common.randomness import configure_torch
        configure_torch(config)

    if args.command in ("matrix", "preflight"):
        counts = expected_matrix_counts(config) if not _has_filter(args) and args.shard_count == 1 else {
            "conditions": len(selected),
            "training_runs": sum(condition.trainable for condition in selected),
            "static_states": sum(not condition.trainable for condition in selected),
            "routed_states": sum(config["training"]["epochs"] + 1 if condition.trainable else 1 for condition in selected),
        }
        print(json.dumps({"suite": config["suite"]["name"], **counts}, indent=2))
        if args.list:
            for condition in selected:
                print(json.dumps(condition.to_dict(), sort_keys=True))
        if args.command == "preflight" and not args.config_only:
            from task6_phaseb.substrate.assets import inspect_task
            for task, experts in sorted({(c.task, c.experts) for c in selected}):
                _, _, _, identity = inspect_task(config, task, experts)
                print(json.dumps({"task": task, "experts": experts, "identity": identity}, sort_keys=True))
        return

    if args.command == "import-e64":
        from task6_phaseb.imports.phase_a_f0 import import_phase_a_f0
        import_phase_a_f0(config, args.run_id)
    elif args.command in ("prepare", "validate"):
        if args.shard_count != 1 or any(getattr(args, key) is not None for key in ("arm", "variant", "k", "seed")):
            raise ValueError("Preparation and validation may only be filtered by task/E")
        for task, experts in sorted({(c.task, c.experts) for c in selected}):
            if args.command == "prepare":
                from task6_phaseb.common.context import prepare_task
                prepare_task(config, task, experts, args.run_id)
            else:
                from task6_phaseb.substrate.validation import validate_task
                validate_task(config, task, experts, args.run_id)
    elif args.command == "train":
        from task6_phaseb.training.runner import train_condition
        if args.resume and len(selected) != 1:
            raise ValueError("--resume requires one condition")
        for condition in selected:
            train_condition(config, condition, args.run_id, args.resume, args.stop_after_epoch)
    elif args.command == "capture":
        from task6_phaseb.capture.runner import capture_diagnostics, capture_predictions
        from task6_phaseb.metrics.performance.pipeline import select_best
        if args.part in ("all", "A"):
            for condition in selected:
                capture_predictions(config, condition, args.run_id, args.skip_complete)
        if args.part in ("all", "select-best"):
            for condition in selected:
                select_best(config, condition, args.run_id)
        if args.part in ("all", "diagnostics"):
            for condition in selected:
                capture_diagnostics(config, condition, args.run_id, args.skip_complete)
    elif args.command == "metrics":
        from task6_phaseb.metrics.pipeline import compute_condition
        for condition in selected:
            compute_condition(config, condition, args.run_id, args.metric)
    elif args.command == "aggregate":
        from task6_phaseb.aggregation.pipeline import aggregate
        aggregate(config, args.run_id)
    elif args.command in ("tables", "figures"):
        from task6_phaseb.visualization.render import figures, tables
        (tables if args.command == "tables" else figures)(config, args.run_id)
    elif args.command == "smoke":
        if config["suite"]["name"] != "smoke":
            raise ValueError("smoke requires configs/suites/smoke.yaml")
        common = ["--suite", args.suite, "--run-id", args.run_id]
        if args.local:
            common += ["--local", args.local]
        # Smoke validates the model/data path on one E/k/seed. Combined E64/E128/E256
        # aggregation and figures are formal-suite completeness checks.
        for command in ("import-e64", "prepare", "validate", "train", "capture", "metrics"):
            main([command, *common])


if __name__ == "__main__":
    main()
