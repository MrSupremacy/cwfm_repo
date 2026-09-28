from __future__ import annotations

import argparse
import json

from task5.common.config import ARMS, conditions, extension_arms, extension_spec, load_config, validate_run_id


def parser():
    p = argparse.ArgumentParser(description="Task 5: independent train -> capture -> metrics -> figures")
    p.add_argument("command", choices=["matrix", "preflight", "prepare", "validate", "train", "capture", "metrics", "aggregate", "tables", "figures", "smoke", "phase-a-check", "phase-a-report"])
    p.add_argument("--suite", help="YAML suite, default configs/suites/main.yaml")
    p.add_argument("--local", help="Server-only input paths and device overrides")
    p.add_argument("--run-id", default="main01", help="Explicit run family; never automatically select latest outputs")
    p.add_argument("--task", choices=["sst2", "mnli"])
    p.add_argument("--arm", choices=["dense", *ARMS])
    p.add_argument("--variant")
    p.add_argument("--k", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1, help="Manual independent-run partitioning, not DDP")
    p.add_argument("--part", choices=["A", "select-best", "diagnostics", "E", "all"], default="all")
    p.add_argument("--metric", choices=["all", "performance", "load_balance", "churn", "oracle_overlap", "activation_coverage", "coactivation_consistency"], default="all")
    p.add_argument("--resume", help="Checkpoint directory name within the selected single training run")
    p.add_argument("--stop-after-epoch", type=int, help="Deliberate epoch-boundary stop for resume verification")
    p.add_argument("--skip-complete", action="store_true", help="Verify and reuse already complete capture outputs")
    p.add_argument("--config-only", action="store_true", help="Preflight without accessing remote inputs")
    p.add_argument("--list", action="store_true", help="List matrix conditions")
    return p


def select_conditions(config, args):
    selected = conditions(config)
    extension = config.get("extension")
    if extension and "arms" in extension and args.command in ("matrix", "preflight", "train", "capture", "metrics"):
        selected = [c for c in selected if c.arm in extension_arms(extension)]
    for key in ("task", "arm", "variant", "k", "seed"):
        value = getattr(args, key)
        if value is not None:
            selected = [c for c in selected if getattr(c, key) == value]
    if args.command == "train":
        selected = [c for c in selected if c.trainable]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    selected = selected[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError("No matching experiment conditions")
    return selected


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run_id(args.run_id)
    config = load_config(args.suite, args.local)
    selected = select_conditions(config, args)
    extension = extension_spec(config, args.run_id)
    if extension is not None:
        if args.command in ("prepare", "validate", "smoke"):
            raise ValueError("Extension mode reuses an immutable prepared/validated base; do not rerun this stage")
        if args.command in ("train", "capture", "metrics") and any(c.arm not in extension_arms(extension) for c in selected):
            raise ValueError(f"Extension model/data stages require arms {extension_arms(extension)}")
        if "arms" in extension and args.command in ("aggregate", "tables", "figures"):
            raise ValueError("Use phase-a-report for the isolated F0 comparison; old results are read-only")
        if args.command == "capture" and args.part in ("E", "all"):
            raise ValueError("Extension mode reuses the base E capture; run A, select-best, and diagnostics separately")
    model_stages = {"prepare", "validate", "train", "smoke"}
    needs_model = args.command in model_stages or (args.command == "capture" and args.part != "select-best")
    if needs_model:
        from task5.common.randomness import configure_torch
        configure_torch(config)
    if args.command in ("matrix", "preflight"):
        count_train = sum(c.trainable for c in selected)
        count_static = len(selected)-count_train
        print(json.dumps({"suite": config["suite"]["name"], "conditions": len(selected), "training_runs": count_train,
                          "static_states_including_dense": count_static,
                          "total_logical_states": count_train*(config["training"]["epochs"]+1)+count_static}, indent=2))
        if args.list:
            for c in selected:
                print(json.dumps(c.to_dict()))
        if args.command == "preflight" and not args.config_only:
            from task5.substrate.assets import inspect_task
            for task in sorted({c.task for c in selected}):
                _, _, identity = inspect_task(config, task)
                print(task, json.dumps(identity))
                if extension is not None:
                    from task5.common.context import verify_prepared
                    prepared = verify_prepared(config, task, args.run_id)
                    print(task, json.dumps({"reused_prepared_protocol": prepared["protocol"]}))
        return
    if args.command in ("aggregate", "tables", "figures", "smoke", "phase-a-check", "phase-a-report") and (args.shard_count != 1 or any(
            getattr(args, key) is not None for key in ("task", "arm", "variant", "k", "seed"))):
        raise ValueError("This stage requires the complete suite; subset via a separately named suite, not silent filtering")
    if args.command in ("phase-a-check", "phase-a-report"):
        from task5.aggregation.phase_a import check_base, report
        (check_base if args.command == "phase-a-check" else report)(config, args.run_id)
    elif args.command in ("prepare", "validate"):
        if args.shard_count != 1 or any(getattr(args, key) is not None for key in ("arm", "variant", "k", "seed")):
            raise ValueError("Shared preparation/validation partitions by --task only")
        for task in sorted({c.task for c in selected}):
            if args.command == "prepare":
                from task5.common.context import prepare_task
                prepare_task(config, task, args.run_id)
            else:
                from task5.substrate.validation import validate_task
                validate_task(config, task, args.run_id)
    elif args.command == "train":
        from task5.training.runner import train_condition
        if args.resume and len(selected) != 1:
            raise ValueError("--resume requires exactly one condition")
        for c in selected:
            train_condition(config, c, args.run_id, args.resume, args.stop_after_epoch)
    elif args.command == "capture":
        from task5.capture.runner import capture_predictions, capture_diagnostics, capture_coactivation
        from task5.metrics.performance.pipeline import select_best
        if args.part in ("all", "E") and (args.shard_count != 1 or any(getattr(args, key) is not None for key in ("arm", "variant", "k", "seed"))):
            raise ValueError("Shared E must run once per task; shard A/diagnostics separately, then run --part E")
        if args.part in ("all", "A"):
            for c in selected:
                capture_predictions(config, c, args.run_id, args.skip_complete)
        if args.part in ("all", "select-best"):
            for c in selected:
                select_best(config, c, args.run_id)
        if args.part in ("all", "diagnostics"):
            for c in selected:
                capture_diagnostics(config, c, args.run_id, args.skip_complete)
        if args.part in ("all", "E"):
            for task in sorted({c.task for c in selected}):
                capture_coactivation(config, task, args.run_id, args.skip_complete)
    elif args.command == "metrics":
        from task5.metrics.pipeline import compute_condition
        for c in selected:
            compute_condition(config, c, args.run_id, args.metric)
    elif args.command == "aggregate":
        from task5.aggregation.pipeline import aggregate
        aggregate(config, args.run_id)
    elif args.command in ("tables", "figures"):
        from task5.visualization.render import figures, tables
        (tables if args.command == "tables" else figures)(config, args.run_id)
    elif args.command == "smoke":
        if config["suite"]["name"] != "smoke":
            raise ValueError("smoke requires --suite configs/suites/smoke.yaml; never use formal outputs")
        # A convenience sequential test, not a separate orchestration/registry service.
        common = ["--suite", args.suite, "--run-id", args.run_id]
        if args.local:
            common += ["--local", args.local]
        for command in ("prepare", "validate", "train", "capture", "metrics", "aggregate", "tables", "figures"):
            main([command, *common])


if __name__ == "__main__":
    main()
