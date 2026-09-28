#!/usr/bin/env python3
"""Build a safe per-shard plan for resuming the formal training matrix."""

from __future__ import annotations

import argparse
import json
import sys

from task5.common.config import conditions, load_config, protocol_id, run_path, validate_run_id
from task5.common.io import checked_complete, read_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", required=True)
    result.add_argument("--local", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--shard-count", type=int, default=1)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--verify-hashes", action="store_true")
    result.add_argument("--require-complete", action="store_true")
    return result


def inspect_condition(config, condition, run_id, verify_hashes):
    directory = run_path(config, "train", condition, run_id)
    if not directory.exists():
        return "start", "-"
    if not directory.is_dir():
        raise RuntimeError(f"Training path is not a directory: {directory}")

    recorded = read_json(directory / "config.json")
    if recorded["condition"] != condition.to_dict():
        raise ValueError(f"Condition identity mismatch: {directory}")

    checkpoint_dir = directory / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Existing run has no checkpoint directory: {directory}")
    paths = sorted(p for p in checkpoint_dir.iterdir() if not p.name.startswith("."))
    if not paths:
        raise RuntimeError(f"Existing run has no complete checkpoint to resume: {directory}")

    expected_protocol = protocol_id(config)
    states = []
    for path in paths:
        if not path.is_dir():
            raise RuntimeError(f"Unexpected checkpoint entry: {path}")
        meta = read_json(path / "meta.json")
        header = checked_complete(path) if verify_hashes else read_json(path / "complete.json")["header"]
        if header != meta:
            raise ValueError(f"Checkpoint completion header mismatch: {path}")
        if meta["condition"] != condition.to_dict() or meta["protocol"] != expected_protocol:
            raise ValueError(f"Checkpoint condition/protocol mismatch: {path}")
        if meta["name"] != path.name:
            raise ValueError(f"Checkpoint name mismatch: {path}")
        states.append(meta)

    states.sort(key=lambda state: state["epoch"])
    epochs = [state["epoch"] for state in states]
    if epochs != list(range(epochs[-1] + 1)) or epochs[-1] > config["training"]["epochs"]:
        raise ValueError(f"Checkpoint epochs are not a contiguous prefix: {directory} ({epochs})")
    if any(right["step"] <= left["step"] for left, right in zip(states, states[1:])):
        raise ValueError(f"Checkpoint steps are not strictly increasing: {directory}")
    for state in states:
        expected_name = "step_0" if state["epoch"] == 0 else (
            "final" if state["epoch"] == config["training"]["epochs"] else f"step_{state['step']}"
        )
        if state["name"] != expected_name:
            raise ValueError(f"Unexpected checkpoint name for epoch {state['epoch']}: {directory}")

    if epochs[-1] == config["training"]["epochs"]:
        if len(states) != config["training"]["epochs"] + 1:
            raise ValueError(f"Completed run does not contain all checkpoint states: {directory}")
        return "skip", states[-1]["name"]
    return "resume", states[-1]["name"]


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run_id(args.run_id)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    config = load_config(args.suite, args.local)
    selected = [condition for condition in conditions(config) if condition.trainable]
    selected = selected[args.shard_index::args.shard_count]

    counts = {"skip": 0, "resume": 0, "start": 0}
    actions = []
    for condition in selected:
        action, checkpoint = inspect_condition(config, condition, args.run_id, args.verify_hashes)
        counts[action] += 1
        if action != "skip":
            actions.append((action, condition, checkpoint))

    print(json.dumps({"shard_index": args.shard_index, "shard_count": args.shard_count,
                      "selected": len(selected), **counts}, sort_keys=True), file=sys.stderr)
    if args.require_complete and actions:
        raise RuntimeError(f"Formal training matrix is incomplete: {counts}")
    for action, condition, checkpoint in actions:
        print("\t".join((action, condition.task, condition.arm, condition.variant,
                         str(condition.k), str(condition.seed), checkpoint)))


if __name__ == "__main__":
    main()
