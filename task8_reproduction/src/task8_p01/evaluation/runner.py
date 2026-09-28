from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from task8_p01.assets.catalog import load_summaries
from task8_p01.common.config import MODES, TASKS, Snapshot, implementation_id, protocol_id, resolve_path
from task8_p01.common.io import checked_complete, complete, fresh_directory, write_json
from task8_p01.compat.task6 import load_task6_config, task6_source_identity
from task8_p01.data.panel import freeze_panel, panel_members
from task8_p01.substrate.model import attach


def _output_root(config):
    return resolve_path(config, config["execution"]["output_root"])


def result_path(config, population, snapshot, mode, step):
    root = _output_root(config) / "runs/evaluation" / population / f"E_{snapshot.experts}" / f"k_{snapshot.k}"
    if mode == "M00":
        return root / mode / "static"
    if snapshot.role == "init":
        return root / mode / "init" / "step_0"
    return root / mode / "best" / f"seed_{snapshot.seed}" / f"step_{step}"


def _native_metrics(task, gold, prediction):
    gold = np.asarray(gold, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    accuracy = float(np.mean(gold == prediction))
    if task != "qqp":
        return {"accuracy": accuracy, "native": accuracy}
    tp = int(np.sum((gold == 1) & (prediction == 1)))
    fp = int(np.sum((gold == 0) & (prediction == 1)))
    fn = int(np.sum((gold == 1) & (prediction == 0)))
    f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {"accuracy": accuracy, "f1": f1, "native": (accuracy + f1) / 2}


def _candidate_predictions(model, tokenizer, loader, names, device):
    import torch
    import torch.nn.functional as functional

    candidates = []
    for name in names:
        ids = tokenizer(name, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is None:
            raise ValueError("T5 tokenizer requires EOS")
        candidates.append([*ids, tokenizer.eos_token_id])
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            source_ids = batch["source_index"].tolist()
            gold = batch["class_id"].tolist()
            values = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in {"source_index", "class_id", "domain_id", "content_mask"}
            }
            scores = torch.empty((len(source_ids), len(candidates)), dtype=torch.float64)
            for candidate_index, ids in enumerate(candidates):
                labels = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).expand(len(source_ids), -1).contiguous()
                output = model(
                    input_ids=values["input_ids"], attention_mask=values["attention_mask"],
                    labels=labels, use_cache=False,
                )
                log_probs = functional.log_softmax(output.logits.float(), dim=-1)
                token_scores = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                scores[:, candidate_index] = token_scores.mean(1).double().cpu()
            predicted = scores.argmax(1).tolist()
            for row, source in enumerate(source_ids):
                rows.append({
                    "source_index": int(source), "gold": int(gold[row]),
                    "prediction": int(predicted[row]), "candidate_scores": scores[row].tolist(),
                    "correct": bool(predicted[row] == gold[row]),
                })
    return rows


def _datasets(config, task6_config, tokenizer, population, dense_identity):
    from task6_phased.data.datasets import load_raw, tokenize

    result = {}
    for task in TASKS:
        _, validation = load_raw(task6_config, task)
        encoded = tokenize(task6_config, task, validation, tokenizer, "validation", dense_identity)
        panel_hash = None
        if population == "diagnostic_128":
            members, panel_hash = panel_members(config, task)
            encoded = encoded.select(members)
        result[task] = (encoded, panel_hash)
    return result


def _evaluate_mode(config, task6_config, model, tokenizer, controller, datasets, mode):
    from task6_phased.data.datasets import make_loader

    controller.set_mode(mode)
    controller.generation()
    tasks = {}
    for task in TASKS:
        encoded, panel_hash = datasets[task]
        loader = make_loader(
            task6_config,
            encoded,
            tokenizer,
            int(config["evaluation"]["candidate_batch_size"]),
        )
        rows = _candidate_predictions(
            model, tokenizer, loader, task6_config["tasks"][task]["labels"],
            task6_config["execution"]["device"],
        )
        source_ids = [int(value) for value in encoded["source_index"]]
        if [row["source_index"] for row in rows] != source_ids:
            raise ValueError(f"{task}: evaluation lost or reordered examples")
        metrics = _native_metrics(task, [row["gold"] for row in rows], [row["prediction"] for row in rows])
        metrics.update({"count": len(rows), "panel_hash": panel_hash})
        tasks[task] = {"metrics": metrics, "predictions": rows}
    native = [tasks[task]["metrics"]["native"] for task in TASKS]
    return {"tasks": tasks, "summary": {"macro": float(np.mean(native)), "worst_domain": float(np.min(native))}}


def evaluate_snapshot(config, snapshot, population="diagnostic_128", modes=MODES, skip_complete=True):
    if not isinstance(snapshot, Snapshot):
        snapshot = Snapshot(**snapshot)
    modes = tuple(modes)
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError("modes must be a non-empty subset of M00/M01/M10/M11")
    if population not in ("diagnostic_128", "full_validation"):
        raise ValueError(population)
    if population == "diagnostic_128":
        freeze_panel(config)

    task6_config = load_task6_config(config)
    from task6_phased.dense.model import load_t5
    from task6_phased.substrate.assets import inspect_assets

    paths, labels, c0, asset_identity = inspect_assets(task6_config, snapshot.experts)
    learned, checkpoint = load_summaries(config, snapshot)
    model, tokenizer = load_t5(task6_config, paths["dense_best"], trainable=False)
    controller = attach(config, model, labels, c0, learned, snapshot.k)
    datasets = _datasets(
        config, task6_config, tokenizer, population, asset_identity["dense"]["model_sha256"]
    )
    source_identity = task6_source_identity(config)["sha256"]
    outputs = {}
    for mode in modes:
        path = result_path(config, population, snapshot, mode, checkpoint["step"])
        header = {
            "schema": 1, "protocol": protocol_id(config), "population": population,
            "experts": snapshot.experts, "k": snapshot.k, "mode": mode,
            "seed": None if mode == "M00" else snapshot.seed,
            "checkpoint_role": "static" if mode == "M00" else snapshot.role,
            "checkpoint_step": 0 if mode == "M00" else checkpoint["step"],
            "checkpoint_hash": None if mode == "M00" else checkpoint["state_sha256"],
            "checkpoint_summary_hash": None if mode == "M00" else checkpoint["summary_sha256"],
            "dense_hash": asset_identity["dense"]["model_sha256"],
            "split_hash": asset_identity["split"]["files"],
            "task8_source_hash": implementation_id(),
            "task6_source_hash": source_identity,
        }
        if path.exists():
            if not skip_complete:
                raise FileExistsError(path)
            checked_complete(path, header)
            outputs[mode] = path
            continue
        value = _evaluate_mode(config, task6_config, model, tokenizer, controller, datasets, mode)
        with fresh_directory(path):
            for task in TASKS:
                with gzip.open(path / f"{task}.json.gz", "wt", encoding="utf-8") as stream:
                    json.dump(value["tasks"][task], stream, ensure_ascii=False, allow_nan=False)
            write_json(path / "summary.json", value["summary"])
            write_json(path / "identity.json", header)
            complete(path, header)
        outputs[mode] = path
    return outputs
