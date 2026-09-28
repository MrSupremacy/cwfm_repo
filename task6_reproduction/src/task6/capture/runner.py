from __future__ import annotations

import numpy as np

from task6.capture.storage import ProbeWriter, write_parquet
from task6.common.config import protocol_id, run_path
from task6.common.context import load_context
from task6.common.io import checked_complete, complete, fresh_output, terminal_log
from task6.data.datasets import make_loader, move_batch
from task6.training.checkpoints import load_for_capture, states


def capture_path(config, condition, run_id, state, kind):
    population = "probe" if kind == "probe" else "validation"
    return run_path(config, f"capture/{population}", condition, run_id) / state["name"] / kind


def header_for(config, condition, state, input_info, kind, with_q=False):
    return {
        "schema": 1, "protocol": protocol_id(config), "condition": condition.to_dict(),
        "state": state, "input_header": input_info, "kind": kind, "with_q": with_q,
    }


def reuse(path, header, skip_complete):
    if skip_complete and path.exists():
        checked_complete(path, header)
        print(f"Verified existing capture: {path}")
        return True
    return False


def normalize_prediction(text):
    return " ".join(text.strip().lower().split())


def capture_predictions(config, condition, run_id, skip_complete=False):
    import torch

    model, tokenizer, controller, data, inputs, _ = load_context(config, condition, run_id)
    capture = config["capture"]
    loader = make_loader(config, data, tokenizer, capture["generation_batch_size"])
    labels = config["tasks"][condition.task]["labels"]
    for state in states(config, condition, run_id):
        path = capture_path(config, condition, run_id, state, "A")
        header = header_for(config, condition, state, inputs, "A")
        if reuse(path, header, skip_complete):
            continue
        load_for_capture(config, condition, run_id, state, model, controller, inputs)
        model.eval()
        controller.generation()
        records = {key: [] for key in ("sample_id", "prediction", "prediction_right", "prediction_valid")}
        with fresh_output(path), terminal_log(path / "logs/capture.log"), torch.no_grad():
            for batch in loader:
                values = move_batch(batch, config["execution"]["device"])
                output = model.generate(
                    input_ids=values["input_ids"], attention_mask=values["attention_mask"],
                    max_new_tokens=capture["max_new_tokens"], num_beams=capture["num_beams"],
                    do_sample=capture["do_sample"],
                )
                decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                if len(decoded) != len(batch["sample_id"]):
                    raise ValueError("Generation lost samples")
                for sample, prediction, gold in zip(
                        batch["sample_id"].tolist(), decoded, batch["class_id"].tolist()):
                    parsed = normalize_prediction(prediction)
                    records["sample_id"].append(sample)
                    records["prediction"].append(prediction)
                    records["prediction_right"].append(parsed == labels[gold])
                    records["prediction_valid"].append(parsed in labels)
            if records["sample_id"] != list(data["sample_id"]):
                raise ValueError("A does not cover complete validation in original order")
            write_parquet(path / "predictions.parquet", records, "A",
                          compression_level=capture["compression_level"])
            complete(path, header)
            print(f"Captured {len(records['sample_id'])} predictions: {condition} / {state['name']}")


def capture_diagnostics(config, condition, run_id, skip_complete=False):
    import torch

    if not condition.is_routed:
        return
    model, tokenizer, controller, data, inputs, members = load_context(config, condition, run_id)
    capture, experts = config["capture"], config["model"]["num_experts"]
    from task6.metrics.performance.pipeline import best_state
    best = best_state(config, condition, run_id)["name"]
    probe = data.select(members["sample_ids"])

    for state in states(config, condition, run_id):
        modes = ["probe"]
        if state["name"] in (best, "final"):
            modes.insert(0, "B")
        load_for_capture(config, condition, run_id, state, model, controller, inputs)
        model.eval()
        for mode in modes:
            with_q = mode == "probe"
            path = capture_path(config, condition, run_id, state, mode)
            header = header_for(config, condition, state, inputs, mode, with_q)
            if reuse(path, header, skip_complete):
                continue
            dataset = probe if mode == "probe" else data
            batch_size = capture["probe_batch_size"] if mode == "probe" else capture["teacher_batch_size"]
            loader = make_loader(config, dataset, tokenizer, batch_size)
            counts = {key: np.zeros(experts, dtype=np.int64) for key in controller.wrappers}
            tokens = dict.fromkeys(controller.wrappers, 0)
            with fresh_output(path), terminal_log(path / "logs/capture.log"), torch.no_grad():
                writers = {
                    key: ProbeWriter(
                        path / key, key, members["expected_keys"][wrapper.stack], condition.k,
                        experts, True, capture["shard_rows"], capture["compression_level"],
                    )
                    for key, wrapper in controller.wrappers.items()
                } if mode == "probe" else {}

                def observe(wrapper, shape, valid, selected, q):
                    selected_np = selected[valid].detach().cpu().numpy()
                    if mode == "B":
                        counts[wrapper.key] += np.bincount(selected_np.reshape(-1), minlength=experts)
                        tokens[wrapper.key] += len(selected_np)
                    else:
                        if q is None:
                            raise ValueError("Every Phase A probe row requires current local q")
                        rows, positions = torch.where(valid.reshape(shape))
                        keys = np.column_stack((sample_ids[rows.cpu().numpy()], positions.cpu().numpy()))
                        writers[wrapper.key].add(keys, selected_np, q[valid].cpu().numpy())

                controller.observe(observe, with_q=with_q)
                expected_tokens = {"encoder": 0, "decoder": 0}
                for batch in loader:
                    sample_ids = batch["sample_id"].numpy()
                    values = move_batch(batch, config["execution"]["device"])
                    controller.teacher_batch(values)
                    expected_tokens["encoder"] += int(values["attention_mask"].sum())
                    expected_tokens["decoder"] += int((values["labels"] != -100).sum())
                    model(**values, use_cache=False)
                controller.observe(None)
                if mode == "B":
                    for key, wrapper in controller.wrappers.items():
                        if tokens[key] != expected_tokens[wrapper.stack] or counts[key].sum() != tokens[key] * condition.k:
                            raise ValueError("B token/assignment conservation failed")
                    keys = list(counts)
                    write_parquet(
                        path / "loads.parquet",
                        {"layer_id": keys, "valid_token_count": [tokens[key] for key in keys],
                         "assignment_counts": np.stack([counts[key] for key in keys])},
                        "B", experts, condition.k, capture["compression_level"],
                    )
                else:
                    for writer in writers.values():
                        writer.finish()
                complete(path, header)
                print(f"Captured {mode}{'+D' if mode == 'probe' else ''}: {condition} / {state['name']}")
