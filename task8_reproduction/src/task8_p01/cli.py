from __future__ import annotations

import argparse
import json

from task8_p01.common.config import MODES, Snapshot, load_config, matrix_counts


COMMANDS = (
    "matrix", "build-catalog", "inventory", "preflight", "p0-properties",
    "freeze-panel", "evaluate", "compare-endpoints", "replay", "aggregate", "figures",
)


def parser():
    value = argparse.ArgumentParser(description="Task 8 P0/P1 no-training interventions")
    value.add_argument("command", choices=COMMANDS)
    value.add_argument("--suite", default="configs/suites/p01_init_best.yaml")
    value.add_argument("--local")
    value.add_argument("--run-id", default="p01")
    value.add_argument("--experts", type=int, choices=(64, 128, 256))
    value.add_argument("--k", type=int)
    value.add_argument("--seed", type=int, choices=(0, 1, 2))
    value.add_argument("--role", choices=("init", "best"), default="best")
    value.add_argument("--mode", choices=(*MODES, "all"), default="all")
    value.add_argument("--population", choices=("diagnostic_128", "full_validation"), default="diagnostic_128")
    value.add_argument("--create-links", action="store_true")
    value.add_argument("--no-skip-complete", action="store_true")
    return value


def _snapshot(args):
    missing = [name for name in ("experts", "k") if getattr(args, name) is None]
    if args.role == "best" and args.seed is None:
        missing.append("seed")
    if missing:
        raise ValueError(f"This command requires: {', '.join('--' + name for name in missing)}")
    if args.role == "init" and args.seed is not None:
        raise ValueError("--role init is seed-deduplicated; omit --seed")
    return Snapshot(args.experts, args.k, args.seed, args.role)


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.suite, args.local)
    if args.command == "matrix":
        print(json.dumps(matrix_counts(), indent=2))
    elif args.command == "build-catalog":
        from task8_p01.assets.catalog import build_catalog
        value = build_catalog(config, create_links=args.create_links)
        print(json.dumps({"entries": len(value["entries"]), "checkpoint_roles": value["checkpoint_roles"]}, indent=2))
    elif args.command in ("inventory", "preflight"):
        from task8_p01.assets.catalog import inventory
        from task8_p01.common.config import resolve_path
        output = resolve_path(config, config["execution"]["output_root"]) / "results" / args.run_id / "tables/p0_asset_inventory.csv"
        value = inventory(config, output)
        print(json.dumps({"assets": len(value["rows"]), "output": str(output)}, indent=2))
    elif args.command == "p0-properties":
        from task8_p01.common.config import resolve_path
        from task8_p01.p0.checks import run_property_checks
        output = resolve_path(config, config["execution"]["output_root"]) / "results" / args.run_id / "tables/p0_implementation_checks.csv"
        rows = run_property_checks(config, output)
        print(json.dumps({"checks": len(rows), "status": "pass", "output": str(output)}, indent=2))
    elif args.command == "freeze-panel":
        from task8_p01.data.panel import freeze_panel
        print(json.dumps(freeze_panel(config), indent=2))
    elif args.command == "evaluate":
        from task8_p01.evaluation.runner import evaluate_snapshot
        modes = MODES if args.mode == "all" else (args.mode,)
        result = evaluate_snapshot(
            config, _snapshot(args), args.population, modes,
            skip_complete=not args.no_skip_complete,
        )
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    elif args.command == "compare-endpoints":
        from task8_p01.p0.reproduction import compare_diagnostic_endpoints
        print(json.dumps(compare_diagnostic_endpoints(config, _snapshot(args), args.run_id), indent=2))
    elif args.command == "replay":
        from task8_p01.replay.runner import run_local_replay
        print(run_local_replay(config, _snapshot(args), skip_complete=not args.no_skip_complete))
    elif args.command == "aggregate":
        from task8_p01.reporting.aggregate import aggregate_results
        print(json.dumps(aggregate_results(config, args.run_id), indent=2))
    elif args.command == "figures":
        from task8_p01.reporting.figures import render_figures
        print(render_figures(config, args.run_id))


if __name__ == "__main__":
    main()
