from __future__ import annotations

import argparse
import json
from pathlib import Path

from task6_phased.common.config import (
    ARMS, conditions, expected_matrix_counts, load_config, suite_experts, validate_run_id,
)


COMMANDS = (
    "matrix", "preflight-dense", "batch-smoke-dense", "download-data", "train-dense", "evaluate-dense",
    "select-dense-best", "dense-results", "export-dense-best",
    "generate-split", "validate-split", "prepare-routed", "preflight-routed",
    "train-router", "capture", "metrics", "aggregate", "tables", "figures",
)
METRICS = ("all", "performance", "load_balance", "churn", "oracle_overlap", "activation_coverage", "specialization")


def parser():
    value = argparse.ArgumentParser(description="Task 6 Phase D: Dense-MT then routed F0")
    value.add_argument("command", choices=COMMANDS)
    value.add_argument("--suite", required=True)
    value.add_argument("--local")
    value.add_argument("--run-id", default="main01")
    value.add_argument("--arm", choices=ARMS)
    value.add_argument("--experts", type=int, choices=(128, 256))
    value.add_argument("--k", type=int)
    value.add_argument("--seed", type=int)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--shard-count", type=int, default=1)
    value.add_argument("--resume")
    value.add_argument("--part", choices=("A", "select-best", "diagnostics", "all"), default="all")
    value.add_argument("--metric", choices=METRICS, default="all")
    value.add_argument("--skip-complete", action="store_true")
    value.add_argument("--list", action="store_true")
    return value


def selected_conditions(config, args):
    result = conditions(config)
    for key in ("experts", "arm", "k", "seed"):
        wanted = getattr(args, key)
        if wanted is not None:
            result = [condition for condition in result if getattr(condition, key) == wanted]
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    result = result[args.shard_index::args.shard_count]
    return result


def _has_condition_filter(args):
    return any(getattr(args, key) is not None for key in ("experts", "arm", "k", "seed"))


def _selected_matrix_counts(config, selected):
    trained = sum(condition.trainable for condition in selected)
    static = len(selected) - trained
    points = (
        int(config["training"]["total_optimizer_steps"])
        // int(config["training"]["checkpoint_every_steps"])
        + 1
    )
    return {
        "conditions": len(selected), "training_runs": trained, "static_states": static,
        "router_checkpoints": trained * points, "routed_states": trained * points + static,
    }


def _configure(config, command):
    if command not in ("matrix", "download-data"):
        from task6_phased.common.randomness import configure_torch
        configure_torch(config)


def _preflight_dense(config):
    from task6_phased.common.config import TASKS, input_root, root_for, tmp_root
    from task6_phased.common.io import sha256
    from task6_phased.common.provenance import environment, path_identity
    from task6_phased.data.datasets import dataset_paths, load_raw
    from task6_phased.dense.model import resolve_asset
    source = resolve_asset(config, "pretrained")
    if not source.is_dir():
        raise FileNotFoundError(source)
    identities = {"pretrained": path_identity(source), "datasets": {}}
    for task in TASKS:
        paths = dataset_paths(config, task)
        train, validation = load_raw(config, task)
        identities["datasets"][task] = {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": {key: sha256(value) for key, value in paths.items()},
            "counts": {"train": len(train), "validation": len(validation)},
        }
    for name, path in (("output", root_for(config)), ("input", input_root(config)), ("tmp", tmp_root(config))):
        if str(path).startswith("/root/"):
            raise ValueError(f"Formal {name} path must use shared storage, not /root")
    return {"environment": environment(), "identities": identities}


def _complete_suite(args):
    if _has_condition_filter(args) or args.shard_count != 1:
        raise ValueError("This stage requires the complete unfiltered suite")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run_id(args.run_id)
    config = load_config(args.suite, args.local)
    _configure(config, args.command)
    selected = selected_conditions(config, args)

    if args.command == "matrix":
        counts = (
            _selected_matrix_counts(config, selected)
            if _has_condition_filter(args) or args.shard_count != 1
            else expected_matrix_counts(config)
        )
        print(json.dumps(counts, indent=2))
        if args.list:
            for condition in selected:
                print(json.dumps(condition.to_dict(), sort_keys=True))
        return
    if args.command == "preflight-dense":
        print(json.dumps(_preflight_dense(config), indent=2))
    elif args.command == "batch-smoke-dense":
        from task6_phased.dense.trainer import batch_smoke_dense
        print(json.dumps(batch_smoke_dense(config), indent=2))
    elif args.command == "download-data":
        from task6_phased.data.download import download_glue
        download_glue(config)
    elif args.command == "train-dense":
        from task6_phased.dense.trainer import train_dense
        train_dense(config, args.run_id, args.resume)
    elif args.command == "evaluate-dense":
        from task6_phased.dense.evaluation import evaluate_dense
        evaluate_dense(config, args.run_id)
    elif args.command == "select-dense-best":
        from task6_phased.dense.evaluation import select_dense_best
        print(json.dumps(select_dense_best(config, args.run_id), indent=2))
    elif args.command == "dense-results":
        from task6_phased.dense.reporting import dense_results
        print(dense_results(config, args.run_id))
    elif args.command == "export-dense-best":
        from task6_phased.dense.export import export_dense_best
        print(json.dumps(export_dense_best(config, args.run_id), indent=2))
    elif args.command == "generate-split":
        from task6_phased.split.balanced_kmeans import generate_split
        for experts in ([args.experts] if args.experts is not None else suite_experts(config)):
            print(generate_split(config, experts))
    elif args.command == "validate-split":
        from task6_phased.split.validation import validate_split
        result = {
            str(experts): validate_split(config, experts)
            for experts in ([args.experts] if args.experts is not None else suite_experts(config))
        }
        print(json.dumps(result, indent=2))
    elif args.command == "prepare-routed":
        from task6_phased.common.context import prepare
        result = {
            str(experts): prepare(config, args.run_id, experts)
            for experts in ([args.experts] if args.experts is not None else suite_experts(config))
        }
        print(json.dumps(result, indent=2))
    elif args.command == "preflight-routed":
        from task6_phased.common.context import verify_prepared
        from task6_phased.split.validation import validate_split
        from task6_phased.substrate.validation import validate_substrate
        experts_values = [args.experts] if args.experts is not None else suite_experts(config)
        print(json.dumps({
            "matrix": expected_matrix_counts(config),
            "experts": {
                str(experts): {
                    "split": validate_split(config, experts),
                    "prepared": verify_prepared(config, args.run_id, experts),
                    "substrate": validate_substrate(config, experts),
                }
                for experts in experts_values
            },
        }, indent=2))
    elif args.command == "train-router":
        from task6_phased.training.runner import train_condition
        trainable = [condition for condition in selected if condition.trainable]
        if args.resume and len(trainable) != 1:
            raise ValueError("--resume requires one selected trainable condition")
        for condition in trainable:
            train_condition(config, condition, args.run_id, args.resume)
    elif args.command == "capture":
        from task6_phased.capture.runner import capture_diagnostics, capture_predictions
        from task6_phased.metrics.performance.pipeline import select_best
        if args.part in ("A", "all"):
            for condition in selected:
                capture_predictions(config, condition, args.run_id, args.skip_complete)
        if args.part in ("select-best", "all"):
            for condition in selected:
                select_best(config, condition, args.run_id)
        if args.part in ("diagnostics", "all"):
            for condition in selected:
                capture_diagnostics(config, condition, args.run_id, args.skip_complete)
    elif args.command == "metrics":
        from task6_phased.metrics.pipeline import compute_condition
        for condition in selected:
            compute_condition(config, condition, args.run_id, args.metric)
    elif args.command == "aggregate":
        _complete_suite(args)
        from task6_phased.aggregation.pipeline import aggregate
        print(aggregate(config, args.run_id))
    elif args.command in ("tables", "figures"):
        _complete_suite(args)
        from task6_phased.visualization.render import figures, tables
        print((tables if args.command == "tables" else figures)(config, args.run_id))


if __name__ == "__main__":
    main()
