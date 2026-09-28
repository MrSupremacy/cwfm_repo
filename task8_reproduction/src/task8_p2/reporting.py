from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from task8_p01.common.config import TASKS, digest
from task8_p01.common.io import checked_complete, read_json, write_csv, write_json
from task8_p2.artifacts import artifact_lock, atomic_artifact, reuse
from task8_p2.config import cell_fields, code_id, result_root
from task8_p2.evaluation import eval_path, p1_endpoint
from task8_p2.local import TABLES, local_path
from task8_p2.progress import Progress, heartbeat, log


def read_rows(path, predicate=None):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if predicate is None or predicate(row):
                yield row


def join_csv(paths, destination):
    """Stream potentially million-row rank tables instead of loading them all."""
    names = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            columns = next(csv.reader(stream))
        for name in columns:
            if name not in names:
                names.append(name)
    with Path(destination).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, names, extrasaction="raise")
        writer.writeheader()
        for path in paths:
            writer.writerows(read_rows(path))


def verify_local(ctx, snapshots):
    paths = []
    for snapshot in snapshots:
        path = local_path(ctx, snapshot)
        header = checked_complete(path)
        if header["protocol"] != ctx.identity()["protocol"] or header["code_hash"] != code_id():
            raise ValueError(f"Local result is from a different implementation: {path}")
        if not read_json(path/"checks.json")["passed"]:
            raise ValueError(f"Local diagnostic checks failed: {path}")
        paths.append(path)
    return paths


def full_rows(ctx, snapshot):
    _, checkpoint = ctx.snapshot(snapshot)
    assets = ctx.assets(snapshot.experts)[3]
    # Smoke references need the same subset as the new evaluations. Load the
    # datasets once in that isolated path, not as a fresh full formal run.
    if ctx.smoke:
        model, tokenizer = ctx.load_model(snapshot.experts)
        datasets = ctx.datasets(tokenizer, "full_validation")
        del model
        references = {1: p1_endpoint(ctx, snapshot, checkpoint, assets, "M11", datasets),
                      "uniform": p1_endpoint(ctx, snapshot, checkpoint, assets, "M10", datasets)}
    else:
        references = {1: p1_endpoint(ctx, snapshot, checkpoint, assets, "M11"),
                      "uniform": p1_endpoint(ctx, snapshot, checkpoint, assets, "M10")}
    conditions = dict(references)
    for multiplier in (2, 4, 8):
        root = eval_path(ctx, snapshot, multiplier, checkpoint)
        identity = checked_complete(root)
        expected = {"protocol": ctx.identity("full_validation")["protocol"], "code_hash": code_id(),
                    "checkpoint_hash": checkpoint["state_sha256"], "checkpoint_step": checkpoint["step"],
                    "seed": snapshot.seed, "E": snapshot.experts, "k": snapshot.k, "multiplier": multiplier}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(f"N3 output identity mismatch: {root}")
        tasks = {}
        for task in TASKS:
            with gzip.open(root/f"{task}.json.gz", "rt", encoding="utf-8") as stream:
                value = json.load(stream)
            tasks[task] = {"metrics": value["metrics"], "prediction_path": str(root/f"{task}.json.gz")}
        conditions[multiplier] = {"tasks": tasks, "summary": read_json(root/"summary.json"),
                                  "identity": identity, "source": "new_P2", "artifact_path": str(root)}
    rows = []
    for multiplier in (1, 2, 4, 8, "uniform"):
        condition = conditions[multiplier]
        header = condition["identity"]
        base = {**{key: value for key, value in ctx.identity("full_validation").items() if key != "data_definition"},
                **cell_fields(snapshot.experts, snapshot.k),
                "seed": snapshot.seed, "checkpoint_role": "best", "checkpoint_step": checkpoint["step"],
                "checkpoint_hash": checkpoint["state_sha256"], "multiplier": multiplier,
                "forward_mode": "natural_full_model", "source": condition["source"],
                "source_code_hash": header.get("task8_source_hash", header.get("code_hash")),
                "artifact_path": condition["artifact_path"]}
        for task in (*TASKS, "__macro__", "__worst__"):
            if task.startswith("__"):
                field = "macro" if task == "__macro__" else "worst_domain"
                score = condition["summary"][field]
                old = references[1]["summary"][field]
                uniform = references["uniform"]["summary"][field]
                metrics, path = {"native": score}, str(Path(condition["artifact_path"])/"summary.json")
            else:
                metrics = condition["tasks"][task]["metrics"]
                score = metrics["native"]
                old = references[1]["tasks"][task]["metrics"]["native"]
                uniform = references["uniform"]["tasks"][task]["metrics"]["native"]
                path = condition["tasks"][task]["prediction_path"]
            rows.append({**base, "task": task, **metrics, "delta_to_m1": score-old,
                         "delta_to_uniform": score-uniform, "absolute_distance_to_uniform": abs(score-uniform),
                         "prediction_path": path})
    return rows


def summarize_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["E"], row["k"], row["task"], str(row["multiplier"]))].append(row)
    result = []
    for key, values in sorted(groups.items()):
        base = {name: values[0][name] for name in ("phase", "E", "k", "k_over_E", "budget_tier", "task", "multiplier", "data_role", "data_hash", "code_hash")}
        result.append({**base, "n_seeds": len(values), "seed": "__all__", "checkpoint_role": "best",
                       "native_mean": float(np.mean([r["native"] for r in values])),
                       "native_sample_sd": float(np.std([r["native"] for r in values], ddof=1)) if len(values) > 1 else None,
                       "delta_to_m1_mean": float(np.mean([r["delta_to_m1"] for r in values])),
                       "delta_to_uniform_mean": float(np.mean([r["delta_to_uniform"] for r in values])),
                       "checkpoint_hashes": ";".join(r["checkpoint_hash"] for r in values),
                       "checkpoint_steps": ";".join(str(r["checkpoint_step"]) for r in values)})
    return result


def aggregate(ctx, snapshots, run_id, *, local_only=False, figures=True):
    paths = verify_local(ctx, snapshots)
    identity = {**ctx.identity(), "artifact": "p2_results", "run_id": run_id,
                "snapshots": [s.to_dict() for s in snapshots], "local_only": local_only,
                "local_identities": [read_json(p/"identity.json") for p in paths]}
    root = result_root(ctx.config, run_id)
    with artifact_lock(root):
        if reuse(root, identity):
            return root
        with atomic_artifact(root, identity) as staging:
            tables = staging/"tables"
            tables.mkdir()
            for name in (*TABLES.values(), "p2_local_units_index.csv"):
                log(f"MERGE {name}")
                join_csv([path/name for path in paths], tables/name)
            full = []
            if not local_only:
                progress = Progress("COLLECT N3 full-validation", len(snapshots), ctx.config["p2"]["progress_interval_seconds"])
                for index, snapshot in enumerate(snapshots):
                    full.extend(full_rows(ctx, snapshot))
                    progress.update(index+1)
                write_csv(tables/"p2_n3_full_eval.csv", full)
                summary = summarize_seeds(full)
                write_csv(tables/"p2_n3_seed_summary.csv", summary)
                with (tables/"p2_n3_macro_summary.md").open("w", encoding="utf-8") as stream:
                    stream.write("# P2-N3 macro: seed mean ± sample SD (score points)\n\n")
                    stream.write("| E | k | exact k/E | m | n seeds | macro | Δ vs m1 (pp) | Δ vs uniform (pp) |\n|---:|---:|---:|---|---:|---:|---:|---:|\n")
                    for row in summary:
                        if row["task"] == "__macro__":
                            sd = "N/A" if row["native_sample_sd"] is None else f"{100*row['native_sample_sd']:.2f}"
                            stream.write(f"| {row['E']} | {row['k']} | {100*row['k_over_E']:.5f}% | {row['multiplier']} | {row['n_seeds']} | {100*row['native_mean']:.2f} ± {sd} | {100*row['delta_to_m1_mean']:+.2f} | {100*row['delta_to_uniform_mean']:+.2f} |\n")
            write_json(staging/"manifest.json", {"identity": identity, "status": "smoke_complete" if ctx.smoke else ("local_complete" if local_only else "P2_N_complete"),
                       "new_training_runs": 0, "snapshot_count": len(snapshots), "new_full_evaluations": 0 if local_only else 3*len(snapshots),
                       "reused_p1_endpoints": 0 if local_only else 2*len(snapshots), "full_rows": len(full),
                       "rank_variance_ddof": 1, "rank_weighting": "pooled_nonpadding_token_layer_units",
                       "auto_selected_temperature": None, "local_source_paths": [str(p) for p in paths]})
            if figures:
                from task8_p2.figures import render_figures
                with heartbeat("render P2 PNG/PDF figures", ctx.config["p2"]["heartbeat_seconds"]):
                    render_figures(ctx.config, staging, local_only=local_only)
    log(f"P2_RESULTS_COMPLETE {root}")
    return root
