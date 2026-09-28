"""Compare real CUDA epoch-boundary resume against a completed smoke run."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from task5.common.config import Condition, load_config, run_path
from task5.common.io import checked_complete, write_json
from task5.training.checkpoints import states


def assert_same(left, right, path="state"):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"Resume differs at {path}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"Resume differs at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Resume keys differ at {path}")
        for key in left:
            assert_same(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Resume length differs at {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}/{i}")
    elif left != right:
        raise AssertionError(f"Resume differs at {path}: {left!r} != {right!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/suites/smoke.yaml")
    parser.add_argument("--local", default="configs/local/server.yaml")
    parser.add_argument("--reference-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", choices=["sst2", "mnli"], default="mnli")
    parser.add_argument("--report", default="tmp/server-tests/resume.json")
    args = parser.parse_args()
    config = load_config(args.suite, args.local)
    if config["suite"]["name"] != "smoke" or args.run_id == args.reference_run_id:
        raise ValueError("Use a separate run-id and the smoke suite")
    if config["suite"]["top_k"] != [13] or config["suite"]["seeds"] != [0]:
        raise ValueError("Resume check expects the standard k13/seed0 smoke suite")
    matrix = [Condition(args.task, arm, "default", 13, 0) for arm in ("R4", "G3", "G4")]
    for condition in matrix:
        states(config, condition, args.reference_run_id)
        if run_path(config, "train", condition, args.run_id).exists():
            raise FileExistsError("Resume check refuses to replace previous test outputs")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    common = ["--suite", args.suite, "--local", args.local, "--run-id", args.run_id, "--task", args.task]

    def run(command, *extra):
        subprocess.run([sys.executable, "-u", "-m", "task5", command, *common, *extra], cwd=ROOT, env=env, check=True)

    run("prepare")
    report = {"scope": "real-asset smoke CUDA resume: stop at epoch5 and restart in a fresh process", "arms": []}
    import torch
    for condition in matrix:
        run("train", "--arm", condition.arm, "--stop-after-epoch", "5")
        run("train", "--arm", condition.arm, "--resume", "step_10")
        states(config, condition, args.run_id)
        compared = []
        for name in ("step_10", "final"):
            paths = [run_path(config, "train", condition, run_id) / "checkpoints" / name
                     for run_id in (args.reference_run_id, args.run_id)]
            for path in paths:
                checked_complete(path)
            # Only this project's trusted, checksum-verified local checkpoints.
            expected, resumed = [torch.load(p / "state.pt", map_location="cpu", weights_only=False) for p in paths]
            assert_same(expected, resumed)
            compared.append(name)
        report["arms"].append({"arm": condition.arm, "checkpoints": compared,
                               "routers_optimizer_scheduler_and_all_rng_exactly_equal": True})
        write_json(args.report, report)
        print(f"Exact resume match: {condition.arm}", flush=True)
    report["passed"] = True
    write_json(args.report, report)


if __name__ == "__main__":
    main()
