from __future__ import annotations

from pathlib import Path

import numpy as np

from task6_phased.common.config import TASKS, dense_root, dense_run_path, protocol_id
from task6_phased.common.io import read_json, write_json
from task6_phased.data.datasets import load_raw, make_loader, move_model_batch, tokenize
from task6_phased.dense.checkpoints import dense_states
from task6_phased.dense.model import load_t5


def normalize_text(text):
    return " ".join(text.strip().lower().split())


def _candidate_ids(tokenizer, names):
    result = []
    for name in names:
        ids = tokenizer(name, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is None:
            raise ValueError("T5 tokenizer requires EOS")
        ids = [*ids, tokenizer.eos_token_id]
        if not ids:
            raise ValueError("Empty label candidate")
        result.append(ids)
    return result


def candidate_predictions(model, tokenizer, loader, names, device):
    import torch
    import torch.nn.functional as functional
    candidates = _candidate_ids(tokenizer, names)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            source_ids = batch["source_index"].tolist()
            gold = batch["class_id"].tolist()
            values = move_model_batch(batch, device)
            batch_size = len(source_ids)
            scores = torch.empty((batch_size, len(candidates)), dtype=torch.float64)
            for candidate_index, ids in enumerate(candidates):
                labels = (
                    torch.tensor(ids, dtype=torch.long, device=device)
                    .unsqueeze(0).expand(batch_size, -1).contiguous()
                )
                output = model(
                    input_ids=values["input_ids"], attention_mask=values["attention_mask"],
                    labels=labels, use_cache=False,
                )
                log_probs = functional.log_softmax(output.logits.float(), dim=-1)
                token = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                scores[:, candidate_index] = token.mean(1).double().cpu()
            predicted = scores.argmax(1).tolist()
            for index, source in enumerate(source_ids):
                rows.append({
                    "source_index": int(source), "gold": int(gold[index]),
                    "prediction": int(predicted[index]),
                    "candidate_scores": scores[index].tolist(),
                    "correct": bool(predicted[index] == gold[index]),
                })
    return rows


def greedy_predictions(model, tokenizer, loader, names, device, capture):
    import torch
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            values = move_model_batch(batch, device)
            output = model.generate(
                input_ids=values["input_ids"], attention_mask=values["attention_mask"],
                max_new_tokens=capture["max_new_tokens"], num_beams=capture["num_beams"],
                do_sample=capture["do_sample"],
            )
            text = tokenizer.batch_decode(output, skip_special_tokens=True)
            normalized = [normalize_text(item) for item in text]
            for source, gold, raw, parsed in zip(
                batch["source_index"].tolist(), batch["class_id"].tolist(), text, normalized
            ):
                rows.append({
                    "source_index": int(source), "gold": int(gold), "text": raw,
                    "normalized": parsed, "valid": parsed in names,
                    "exact": parsed == names[int(gold)],
                })
    return rows


def native_metrics(task, gold, prediction):
    gold = np.asarray(gold, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    accuracy = float(np.mean(gold == prediction))
    if task in ("sst2", "mnli", "qnli"):
        return {"accuracy": accuracy, "native": accuracy}
    if task == "qqp":
        tp = int(np.sum((gold == 1) & (prediction == 1)))
        fp = int(np.sum((gold == 0) & (prediction == 1)))
        fn = int(np.sum((gold == 1) & (prediction == 0)))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        return {"accuracy": accuracy, "f1": f1, "native": (accuracy + f1) / 2}
    raise ValueError(task)


def evaluate_model(config, model, tokenizer):
    result = {}
    for task in TASKS:
        _, validation = load_raw(config, task)
        encoded = tokenize(config, task, validation, tokenizer, "validation", {"count": len(validation)})
        candidate_loader = make_loader(config, encoded, tokenizer, config["capture"]["candidate_batch_size"])
        candidate = candidate_predictions(
            model, tokenizer, candidate_loader, config["tasks"][task]["labels"],
            config["execution"]["device"],
        )
        greedy_loader = make_loader(config, encoded, tokenizer, config["capture"]["generation_batch_size"])
        greedy = greedy_predictions(
            model, tokenizer, greedy_loader, config["tasks"][task]["labels"],
            config["execution"]["device"], config["capture"],
        )
        expected = list(range(len(encoded)))
        if [row["source_index"] for row in candidate] != expected or [row["source_index"] for row in greedy] != expected:
            raise ValueError(f"{task}: evaluation lost/reordered examples")
        metrics = native_metrics(
            task, [row["gold"] for row in candidate], [row["prediction"] for row in candidate]
        )
        metrics.update({
            "count": len(candidate),
            "greedy_exact": float(np.mean([row["exact"] for row in greedy])),
            "greedy_invalid_rate": float(np.mean([not row["valid"] for row in greedy])),
        })
        result[task] = {"metrics": metrics, "candidate": candidate, "greedy": greedy}
    native = [result[task]["metrics"]["native"] for task in TASKS]
    result["summary"] = {"macro": float(np.mean(native)), "worst_domain": float(np.min(native))}
    return result


def evaluate_dense(config, run_id):
    train = dense_run_path(config, "train", run_id)
    states = dense_states(
        train / "checkpoints", config["dense"]["total_optimizer_steps"],
        config["dense"]["checkpoint_every_steps"],
        {"protocol": protocol_id(config), "run_id": run_id,
         "total_steps": config["dense"]["total_optimizer_steps"]},
    )
    output = dense_run_path(config, "evaluation", run_id)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": protocol_id(config), "run_id": run_id, "states": []}
    for state in states:
        model, tokenizer = load_t5(config, state["path"] / "model", trainable=False)
        result = evaluate_model(config, model, tokenizer)
        write_json(output / f"{state['name']}.json", result)
        manifest["states"].append({
            "name": state["name"], "step": state["step"],
            "macro": result["summary"]["macro"], "worst_domain": result["summary"]["worst_domain"],
        })
    write_json(output / "manifest.json", manifest)
    return manifest


def select_dense_best(config, run_id):
    directory = dense_run_path(config, "evaluation", run_id)
    manifest = read_json(directory / "manifest.json")
    states = manifest["states"]
    if len(states) != 11 and config["suite"]["name"] != "smoke":
        raise ValueError("Dense best requires all 11 evaluation states")
    selected = min(states, key=lambda row: (-row["macro"], row["step"]))
    result = {
        "protocol": protocol_id(config), "run_id": run_id,
        "selection": "native_macro_equal_domain_best",
        "tie_break": "earlier_step", "state": selected, "candidates": states,
    }
    write_json(directory / "selection.json", result)
    return result
