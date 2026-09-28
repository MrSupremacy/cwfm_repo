"""Verify completed real-data smoke artifacts; do not run training or forwards."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from task5.common.config import conditions, implementation_id, load_config, root_for
from task5.common.context import shared_path
from task5.common.io import checked_complete, read_json, sha256, write_json
from task5.capture.runner import capture_path
from task5.metrics.performance.pipeline import best_state, captured_states
from task5.training.checkpoints import states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/suites/smoke.yaml")
    parser.add_argument("--local", default="configs/local/server.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verify-source", action="store_true", help="Also hash original Task 4 inputs on this server")
    parser.add_argument("--report", default="tmp/server-tests/smoke_audit.json")
    args = parser.parse_args()
    config = load_config(args.suite, args.local)
    if config["suite"]["name"] != "smoke":
        raise ValueError("This audit describes the small smoke suite, not a completed formal experiment")
    root = root_for(config)
    counts = dict(training_runs=0, checkpoints=0, A=0, B=0, probe=0, probe_with_q=0, E_layers=0)
    best = []
    for condition in conditions(config):
        available = states(config, condition, args.run_id)
        if captured_states(config, condition, args.run_id) != available:
            raise ValueError("Captured candidates do not match the saved states")
        if condition.trainable:
            counts["training_runs"] += 1
            counts["checkpoints"] += len(available)
            from task5.common.config import run_path
            for state in available:
                checked_complete(run_path(config, "train", condition, args.run_id) / "checkpoints" / state["name"], state)
            chosen = best_state(config, condition, args.run_id)
            best.append({**condition.to_dict(), "name": chosen["name"], "epoch": chosen["epoch"]})
        else:
            chosen = {"name": "static"}
        for state in available:
            checked_complete(capture_path(config, condition, args.run_id, state, "A"))
            counts["A"] += 1
            if condition.arm == "dense":
                continue
            probe = capture_path(config, condition, args.run_id, state, "probe")
            header = checked_complete(probe)
            counts["probe"] += 1
            counts["probe_with_q"] += int(header["with_q"])
            if state["name"] in (chosen["name"], "final", "static"):
                checked_complete(capture_path(config, condition, args.run_id, state, "B"))
                counts["B"] += 1
    phase0 = {}
    for task in config["suite"]["tasks"]:
        phase = read_json(root / "runs/validate" / task / args.run_id / "phase0.json")
        phase0[task] = {"conditions": len(phase["results"]),
                        "max_force_all_error": max(r["force_all_max_abs"] for r in phase["results"])}
        e_path = shared_path(config, "coactivation", task, args.run_id)
        checked_complete(e_path)
        matrices = list(e_path.glob("*.npz"))
        if len(matrices) != config["model"]["encoder_layers"] + config["model"]["decoder_layers"]:
            raise ValueError("Incomplete E layer coverage")
        counts["E_layers"] += len(matrices)
    provenance = read_json(ROOT / "inputs/provenance.json")
    for name, record in provenance["files"].items():
        if sha256(ROOT / "inputs" / name) != record["sha256"]:
            raise ValueError(f"Copied input changed: {name}")
        if args.verify_source and sha256(record["source"]) != record["sha256"]:
            raise ValueError(f"Original input changed: {name}")
    normalized = read_json(root / "results/data/normalized" / args.run_id / "metrics.json")
    aggregated = read_json(root / "results/data/aggregated" / args.run_id / "metrics.json")
    figure_files = sorted(p for p in (root / "results/figures").rglob("*") if p.is_file() and args.run_id in p.parts)
    png = [p for p in figure_files if p.suffix == ".png"]
    pdf = [p for p in figure_files if p.suffix == ".pdf"]
    if not png or {p.with_suffix("") for p in png} != {p.with_suffix("") for p in pdf}:
        raise ValueError("Expected paired PNG/PDF figures")
    for path in png:
        with path.open("rb") as stream:
            header = stream.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or min(struct.unpack(">II", header[16:24])) < 100:
            raise ValueError(f"Invalid PNG: {path}")
    for path in pdf:
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError(f"Invalid PDF: {path}")
    table_files = sorted(p for p in (root / "results/tables").rglob("*") if p.is_file() and args.run_id in p.parts)
    if len(table_files) < 5 or any(p.stat().st_size == 0 for p in table_files):
        raise ValueError("Missing or empty smoke tables")
    report = {"utc": datetime.now(timezone.utc).isoformat(), "implementation": implementation_id(),
              "run_id": args.run_id, "scope": "512 train / 32 validation per task, k13, seed0; not formal results",
              "counts": counts, "phase0": phase0, "best_states": best,
              "normalized_rows": len(normalized["rows"]), "aggregated_rows": len(aggregated["rows"]),
              "png_files": len(png), "pdf_files": len(pdf), "table_files": len(table_files),
              "input_files_verified": len(provenance["files"]), "original_inputs_verified": args.verify_source,
              "passed": True}
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
