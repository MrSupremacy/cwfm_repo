from __future__ import annotations

import argparse
import json

from task6.common.config import ARMS, DENSE_FT_ARM, METRICS, conditions, load_config, validate_run_id


def parser():
    command = ["matrix", "preflight", "prepare", "validate", "train", "capture",
               "metrics", "aggregate", "compare-f0", "dense-extension-report", "tables", "figures", "smoke",
               "phase-a-check", "phase-a-report"]
    result = argparse.ArgumentParser(description="Task 6 Phase A/F1 full-finetuning experiments")
    result.add_argument("command", choices=command)
    result.add_argument("--suite", help="Suite YAML; defaults to the formal Phase A/F1 suite")
    result.add_argument("--local", help="Machine-specific asset and output paths")
    result.add_argument("--run-id", default="main01")
    result.add_argument("--task", choices=["sst2", "mnli"])
    result.add_argument("--arm", choices=["dense", DENSE_FT_ARM, *ARMS])
    result.add_argument("--variant")
    result.add_argument("--k", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--shard-count", type=int, default=1,
                        help="Partition independent conditions; this is not DDP")
    result.add_argument("--part", choices=["A", "select-best", "diagnostics", "all"], default="all")
    result.add_argument("--metric", choices=["all", *METRICS], default="all")
    result.add_argument("--resume", help="Checkpoint name in one selected run")
    result.add_argument("--stop-after-epoch", type=int)
    result.add_argument("--skip-complete", action="store_true")
    result.add_argument("--config-only", action="store_true")
    result.add_argument("--list", action="store_true")
    result.add_argument("--f0-results", help="Task 5 F0 normalized metrics.json (read-only)")
    result.add_argument("--sparse-result-root", help="Completed routed-only Task 6 results/ directory")
    result.add_argument("--sparse-run-id", help="Run ID inside the routed-only result directory")
    return result


def select_conditions(config, args):
    selected = conditions(config)
    for key in ("task", "arm", "variant", "k", "seed"):
        value = getattr(args, key)
        if value is not None:
            selected = [condition for condition in selected if getattr(condition, key) == value]
    if args.command == "train":
        selected = [condition for condition in selected if condition.trainable]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    selected = selected[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError("No matching experiment conditions")
    return selected


def _require_complete_suite(args):
    filtered = args.shard_count != 1 or any(
        getattr(args, key) is not None for key in ("task", "arm", "variant", "k", "seed"))
    if filtered:
        raise ValueError("This stage requires a complete suite; use a separately named suite for subsets")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run_id(args.run_id)
    config = load_config(args.suite, args.local)
    selected = select_conditions(config, args)

    model_stages = {"prepare", "validate", "train", "smoke"}
    if args.command in model_stages or (args.command == "capture" and args.part != "select-best"):
        from task6.common.randomness import configure_torch
        configure_torch(config)

    if args.command in {"matrix", "preflight"}:
        trained = sum(condition.trainable for condition in selected)
        static = len(selected) - trained
        print(json.dumps({
            "suite": config["suite"]["name"], "conditions": len(selected),
            "training_runs": trained, "static_dense_references": static,
            "full_checkpoints": trained * (config["training"]["epochs"] + 1),
            "A_captures": trained * (config["training"]["epochs"] + 1) + static,
        }, indent=2))
        if args.list:
            for condition in selected:
                print(json.dumps(condition.to_dict()))
        if args.command == "preflight" and not args.config_only:
            from task6.substrate.assets import inspect_task
            for task in sorted({condition.task for condition in selected}):
                _, _, identity = inspect_task(config, task)
                print(task, json.dumps(identity))
        return

    if args.command in {"aggregate", "compare-f0", "dense-extension-report", "tables", "figures", "smoke", "phase-a-check", "phase-a-report"}:
        _require_complete_suite(args)

    if args.command in {"prepare", "validate"}:
        if args.shard_count != 1 or any(getattr(args, key) is not None for key in ("arm", "variant", "k", "seed")):
            raise ValueError("Shared preparation and validation may only be partitioned by task")
        for task in sorted({condition.task for condition in selected}):
            if args.command == "prepare":
                from task6.common.context import prepare_task
                prepare_task(config, task, args.run_id)
            else:
                from task6.substrate.validation import validate_task
                validate_task(config, task, args.run_id)
    elif args.command == "train":
        from task6.training.runner import train_condition
        if args.resume and len(selected) != 1:
            raise ValueError("--resume requires exactly one condition")
        for condition in selected:
            train_condition(config, condition, args.run_id, args.resume, args.stop_after_epoch)
    elif args.command == "capture":
        from task6.capture.runner import capture_diagnostics, capture_predictions
        from task6.metrics.performance.pipeline import select_best
        if args.part in {"A", "all"}:
            for condition in selected:
                capture_predictions(config, condition, args.run_id, args.skip_complete)
        if args.part in {"select-best", "all"}:
            for condition in selected:
                if condition.trainable:
                    select_best(config, condition, args.run_id)
        if args.part in {"diagnostics", "all"}:
            for condition in selected:
                capture_diagnostics(config, condition, args.run_id, args.skip_complete)
    elif args.command == "metrics":
        from task6.metrics.pipeline import compute_condition
        for condition in selected:
            compute_condition(config, condition, args.run_id, args.metric)
    elif args.command == "aggregate":
        from task6.aggregation.pipeline import aggregate
        aggregate(config, args.run_id)
    elif args.command == "compare-f0":
        if not args.f0_results:
            raise ValueError("compare-f0 requires --f0-results")
        from task6.comparison.task5_f0 import compare
        compare(config, args.run_id, args.f0_results)
    elif args.command == "dense-extension-report":
        if not args.sparse_result_root or not args.sparse_run_id:
            raise ValueError("dense-extension-report requires --sparse-result-root and --sparse-run-id")
        from task6.comparison.dense_fullft import report
        report(config, args.run_id, args.sparse_result_root, args.sparse_run_id)
    elif args.command in {"tables", "figures"}:
        from task6.visualization.render import figures, tables
        (tables if args.command == "tables" else figures)(config, args.run_id)
    elif args.command in {"phase-a-check", "phase-a-report"}:
        from task6.aggregation.phase_a import check, report
        (check if args.command == "phase-a-check" else report)(config, args.run_id)
    elif args.command == "smoke":
        if config["suite"]["name"] != "smoke":
            raise ValueError("smoke requires configs/suites/smoke.yaml")
        common = ["--suite", args.suite, "--run-id", args.run_id]
        if args.local:
            common += ["--local", args.local]
        for command in ("prepare", "validate", "train", "capture", "metrics",
                        "aggregate", "phase-a-check", "tables", "figures"):
            main([command, *common])


if __name__ == "__main__":
    main()
