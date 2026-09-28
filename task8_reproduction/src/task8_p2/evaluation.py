from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from task8_p01.common.config import TASKS, digest
from task8_p01.common.io import checked_complete, read_json, write_json
from task8_p01.evaluation.runner import _candidate_predictions, _native_metrics, result_path
from task8_p2.artifacts import artifact_lock, atomic_artifact, reuse
from task8_p2.config import cell_fields, output_root
from task8_p2.progress import ProgressLoader, log
from task8_p2.routing import attach_temperature


def eval_path(ctx, snapshot, multiplier, checkpoint):
    scope = "smoke_evaluation" if ctx.smoke else "evaluation"
    return output_root(ctx.config)/"runs"/scope/ctx.identity("full_validation")["protocol"][:16]/snapshot.path/f"m_{multiplier}"


def p1_endpoint(ctx, snapshot, checkpoint, assets, mode, datasets=None):
    if mode not in ("M10", "M11"):
        raise ValueError("Only registered M10/M11 best endpoints may be reused")
    root = result_path(ctx.legacy, "full_validation", snapshot, mode, checkpoint["step"])
    identity = checked_complete(root)
    expected = {"protocol": ctx.config["p2"]["expected_p1_protocol"],
                "task8_source_hash": ctx.config["p2"]["expected_p1_source"],
                "task6_source_hash": ctx.source_identity["sha256"],
                "population": "full_validation", "experts": snapshot.experts, "k": snapshot.k,
                "mode": mode, "seed": snapshot.seed, "checkpoint_role": "best",
                "checkpoint_step": checkpoint["step"], "checkpoint_hash": checkpoint["state_sha256"],
                "checkpoint_summary_hash": checkpoint["summary_sha256"],
                "dense_hash": assets["dense"]["model_sha256"], "split_hash": assets["split"]["files"]}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"P1 endpoint identity mismatch: {root} / {key}")
    tasks = {}
    for task in TASKS:
        with gzip.open(root/f"{task}.json.gz", "rt", encoding="utf-8") as stream:
            value = json.load(stream)
        predictions = value["predictions"]
        # Old evaluator source is pinned; Task6 additionally verifies parquet
        # hashes. When tokenized data is available, check exact source/gold IDs.
        if datasets is not None:
            wanted = {int(row["source_index"]): int(row["class_id"]) for row in datasets[task]}
            by_id = {r["source_index"]: r for r in predictions}
            if len(by_id) != len(predictions):
                raise ValueError("P1 predictions contain duplicated source IDs")
            if ctx.smoke:
                predictions = [by_id[i] for i in wanted]
            elif [r["source_index"] for r in predictions] != list(wanted):
                raise ValueError("P1 evaluation source order/population differs")
            if any(r["gold"] != wanted[r["source_index"]] for r in predictions):
                raise ValueError("P1 gold labels differ from current data")
        expected_count = int(ctx.task6["tasks"][task]["validation_count"])
        if not ctx.smoke and len(predictions) != expected_count:
            raise ValueError(f"P1 endpoint incomplete for {task}")
        computed = _native_metrics(task, [r["gold"] for r in predictions], [r["prediction"] for r in predictions])
        if not ctx.smoke:
            for metric, score in computed.items():
                if abs(score-value["metrics"][metric]) > 1e-12:
                    raise ValueError("P1 stored scores do not match its predictions")
        tasks[task] = {"metrics": {**computed, "count": len(predictions)},
                       "prediction_path": str(root/f"{task}.json.gz"),
                       "population_prediction_hash": digest([(r["source_index"], r["gold"], r["prediction"]) for r in predictions]),
                       "smoke_predictions": predictions if ctx.smoke else None}
    native = [tasks[t]["metrics"]["native"] for t in TASKS]
    summary = {"macro": float(np.mean(native)), "worst_domain": float(np.min(native))}
    if not ctx.smoke:
        old = read_json(root/"summary.json")
        if any(abs(summary[key]-old[key]) > 1e-12 for key in summary):
            raise ValueError("P1 summary differs from rederived task scores")
    return {"identity": identity, "tasks": tasks, "summary": summary,
            "artifact_path": str(root), "source": "reused_P1", "multiplier": 1 if mode == "M11" else "uniform"}


def evaluate_model(ctx, model, tokenizer, controller, datasets, multiplier):
    from task6_phased.data.datasets import make_loader
    controller.set_multiplier(multiplier)
    controller.generation()
    tasks = {}
    for task in TASKS:
        loader = make_loader(ctx.task6, datasets[task], tokenizer, int(ctx.config["evaluation"]["candidate_batch_size"]))
        wrapped = ProgressLoader(loader, f"EVAL m={multiplier} {task}", ctx.config["p2"]["progress_interval_seconds"])
        predictions = _candidate_predictions(model, tokenizer, wrapped, ctx.task6["tasks"][task]["labels"], ctx.device)
        if [r["source_index"] for r in predictions] != [int(i) for i in datasets[task]["source_index"]]:
            raise ValueError("N3 evaluation lost or reordered examples")
        metrics = _native_metrics(task, [r["gold"] for r in predictions], [r["prediction"] for r in predictions])
        tasks[task] = {"metrics": {**metrics, "count": len(predictions)}, "predictions": predictions}
    native = [tasks[t]["metrics"]["native"] for t in TASKS]
    return {"tasks": tasks, "summary": {"macro": float(np.mean(native)), "worst_domain": float(np.min(native))}}


def validate_smoke_endpoint(value, reference, tolerance=.01):
    maximum = 0.0
    for task in TASKS:
        new = value["tasks"][task]["predictions"]
        old = reference["tasks"][task]["smoke_predictions"]
        if [(r["source_index"], r["gold"], r["prediction"]) for r in new] != [(r["source_index"], r["gold"], r["prediction"]) for r in old]:
            raise ValueError(f"P2 m1/uniform endpoint smoke predictions differ from P1: {task}")
        error = float(np.max(np.abs(np.asarray([r["candidate_scores"] for r in new])-np.asarray([r["candidate_scores"] for r in old]))))
        maximum = max(maximum, error)
        if error > tolerance:
            raise ValueError(f"P2 endpoint candidate score error {error} > {tolerance}")
    return {"passed": True, "prediction_mismatches": 0, "max_candidate_score_error": maximum, "tolerance": tolerance}


def full_snapshot(ctx, snapshot):
    learned, checkpoint = ctx.snapshot(snapshot)
    _, labels, c0, assets = ctx.assets(snapshot.experts)
    # Gate the full-validation stage on per-unit local implementation checks.
    from task8_p2.local import local_path
    local = local_path(ctx, snapshot)
    checked_complete(local)
    if not read_json(local/"checks.json")["passed"]:
        raise ValueError("Local diagnostic checks failed; do not enter N3 full validation")
    model, tokenizer = ctx.load_model(snapshot.experts)
    datasets = ctx.datasets(tokenizer, "full_validation")
    references = {1: p1_endpoint(ctx, snapshot, checkpoint, assets, "M11", datasets),
                  "uniform": p1_endpoint(ctx, snapshot, checkpoint, assets, "M10", datasets)}
    controller = attach_temperature(ctx.legacy, model, labels, c0, learned, snapshot.k)
    if ctx.smoke:
        endpoint_checks = {}
        for multiplier in (1, "uniform"):
            actual = evaluate_model(ctx, model, tokenizer, controller, datasets, multiplier)
            endpoint_checks[str(multiplier)] = validate_smoke_endpoint(actual, references[multiplier])
        log(f"ENDPOINT_SMOKE_OK {snapshot} {endpoint_checks}")
    else:
        endpoint_checks = {"source": "P1 identity/predictions verified; m1/uniform are reused, not recomputed"}
    outputs = []
    for multiplier in (2, 4, 8):
        identity = {**ctx.identity("full_validation"), "artifact": "p2_n3_full", **cell_fields(snapshot.experts, snapshot.k),
                    "seed": snapshot.seed, "checkpoint_role": "best", "checkpoint_step": checkpoint["step"],
                    "checkpoint_hash": checkpoint["state_sha256"], "checkpoint_summary_hash": checkpoint["summary_sha256"],
                    "dense_hash": assets["dense"]["model_sha256"], "split_hash": assets["split"]["files"],
                    "multiplier": multiplier, "selector": "original_R4d_unscaled_logits",
                    "evaluation": ctx.config["evaluation"]}
        path = eval_path(ctx, snapshot, multiplier, checkpoint)
        with artifact_lock(path):
            if reuse(path, identity):
                outputs.append(path)
                continue
            with atomic_artifact(path, identity) as staging:
                value = evaluate_model(ctx, model, tokenizer, controller, datasets, multiplier)
                for task in TASKS:
                    with gzip.open(staging/f"{task}.json.gz", "wt", encoding="utf-8") as stream:
                        json.dump(value["tasks"][task], stream, ensure_ascii=False, allow_nan=False)
                write_json(staging/"summary.json", value["summary"])
                write_json(staging/"endpoint_reuse.json", {
                    str(m): {"path": ref["artifact_path"], "identity": ref["identity"], "summary": ref["summary"]}
                    for m, ref in references.items()})
                write_json(staging/"endpoint_checks.json", endpoint_checks)
            outputs.append(path)
            log(f"FULL_COMPLETE {snapshot} m={multiplier} macro={value['summary']['macro']:.6f}")
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
