from __future__ import annotations

import numpy as np

from task5.capture.storage import ProbeWriter, write_parquet
from task5.common.config import R4_FAMILY, protocol_id, run_path
from task5.common.context import input_header, load_context, shared_path
from task5.common.io import checked_complete, complete, fresh_output, read_json, terminal_log, write_json
from task5.data.datasets import make_loader, move_batch
from task5.training.checkpoints import load_for_capture, states


def capture_path(config, condition, run_id, state, kind):
    part = "probe" if kind == "probe" else "validation"
    return run_path(config, f"capture/{part}", condition, run_id) / state["name"] / kind


def header_for(config, condition, state, input_info, kind, with_q=False):
    return {"schema": 1, "protocol": protocol_id(config), "condition": condition.to_dict(), "state": state,
            "input_header": input_info, "kind": kind, "with_q": with_q}


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
    cap = config["capture"]
    loader = make_loader(config, data, tokenizer, cap["generation_batch_size"])
    labels = config["tasks"][condition.task]["labels"]
    for state in states(config, condition, run_id):
        path = capture_path(config, condition, run_id, state, "A")
        header = header_for(config, condition, state, inputs, "A")
        if reuse(path, header, skip_complete):
            continue
        load_for_capture(config, condition, run_id, state, controller, inputs)
        model.eval()
        controller.generation()
        records = {key: [] for key in ("sample_id", "prediction", "prediction_right", "prediction_valid")}
        with fresh_output(path), terminal_log(path / "logs/capture.log"), torch.no_grad():
            for batch in loader:
                x = move_batch(batch, config["execution"]["device"])
                output = model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                        max_new_tokens=cap["max_new_tokens"], num_beams=cap["num_beams"], do_sample=cap["do_sample"])
                decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                if len(decoded) != len(batch["sample_id"]):
                    raise ValueError("Generation lost samples")
                for sample, prediction, gold in zip(batch["sample_id"].tolist(), decoded, batch["class_id"].tolist()):
                    parsed = normalize_prediction(prediction)
                    records["sample_id"].append(sample)
                    records["prediction"].append(prediction)
                    records["prediction_right"].append(parsed == labels[gold])
                    records["prediction_valid"].append(parsed in labels)
            if records["sample_id"] != list(data["sample_id"]):
                raise ValueError("A does not cover the complete validation set in original order")
            write_parquet(path / "predictions.parquet", records, "A", compression_level=cap["compression_level"])
            complete(path, header)
            print(f"Captured {len(records['sample_id'])} predictions: {condition} / {state['name']}")


def capture_diagnostics(config, condition, run_id, skip_complete=False):
    import torch
    if condition.arm == "dense":
        return
    model, tokenizer, controller, data, inputs, members = load_context(config, condition, run_id)
    cap, experts = config["capture"], config["model"]["num_experts"]
    best = "static"
    if condition.trainable:
        from task5.metrics.performance.pipeline import best_state
        best = best_state(config, condition, run_id)["name"]
    probe = data.select(members["sample_ids"])
    for state in states(config, condition, run_id):
        with_q = condition.arm in R4_FAMILY or state["name"] in (best, "final", "static")
        modes = ["probe"]
        if state["name"] in (best, "final", "static"):
            modes.insert(0, "B")
        load_for_capture(config, condition, run_id, state, controller, inputs)
        model.eval()
        for mode in modes:
            path = capture_path(config, condition, run_id, state, mode)
            header = header_for(config, condition, state, inputs, mode, with_q if mode == "probe" else False)
            if reuse(path, header, skip_complete):
                continue
            dataset = probe if mode == "probe" else data
            batch_size = cap["probe_batch_size"] if mode == "probe" else cap["teacher_batch_size"]
            loader = make_loader(config, dataset, tokenizer, batch_size)
            counts = {key: np.zeros(experts, dtype=np.int64) for key in controller.wrappers}
            tokens = dict.fromkeys(controller.wrappers, 0)
            with fresh_output(path), terminal_log(path / "logs/capture.log"), torch.no_grad():
                writers = {key: ProbeWriter(path / key, key, members["expected_keys"][w.stack], condition.k, experts,
                                           with_q, cap["shard_rows"], cap["compression_level"])
                           for key, w in controller.wrappers.items()} if mode == "probe" else {}

                def observe(wrapper, shape, valid, selected, q, activation):
                    s = selected[valid].detach().cpu().numpy()
                    if mode == "B":
                        counts[wrapper.key] += np.bincount(s.reshape(-1), minlength=experts)
                        tokens[wrapper.key] += len(s)
                    else:
                        rows, positions = torch.where(valid.reshape(shape))
                        keys = np.column_stack((sample_ids[rows.cpu().numpy()], positions.cpu().numpy()))
                        writers[wrapper.key].add(keys, s, None if not with_q else q[valid].cpu().numpy())

                controller.observe(observe, with_q=mode == "probe" and with_q)
                expected_tokens = {"encoder": 0, "decoder": 0}
                for batch in loader:
                    sample_ids = batch["sample_id"].numpy()
                    x = move_batch(batch, config["execution"]["device"])
                    controller.teacher_batch(x)
                    expected_tokens["encoder"] += int(x["attention_mask"].sum())
                    expected_tokens["decoder"] += int((x["labels"] != -100).sum())
                    model(**x, use_cache=False)
                controller.observe(None)
                if mode == "B":
                    for key, w in controller.wrappers.items():
                        if tokens[key] != expected_tokens[w.stack] or counts[key].sum() != tokens[key] * condition.k:
                            raise ValueError("B token/assignment count mismatch")
                    keys = list(counts)
                    write_parquet(path / "loads.parquet", {"layer_id": keys, "valid_token_count": [tokens[k] for k in keys],
                                  "assignment_counts": np.stack([counts[k] for k in keys])}, "B", experts, condition.k, cap["compression_level"])
                else:
                    for writer in writers.values():
                        writer.finish()
                complete(path, header)
                print(f"Captured {mode}{'+D' if mode == 'probe' and with_q else ''}: {condition} / {state['name']}")


def capture_coactivation(config, task, run_id, skip_complete=False):
    import torch
    from task5.common.config import Condition
    condition = Condition(task, "dense")
    model, tokenizer, controller, data, inputs, _ = load_context(config, condition, run_id)
    path = shared_path(config, "coactivation", task, run_id)
    header = dict(inputs, kind="E", neuron_order="original", population="full_validation")
    if reuse(path, header, skip_complete):
        return
    cap = config["capture"]
    device = config["execution"]["device"]
    matrices = {key: torch.zeros((config["model"]["d_ff"],) * 2, device=device, dtype=torch.float32) for key in controller.wrappers}
    counts = dict.fromkeys(controller.wrappers, 0)

    def observe(wrapper, shape, valid, selected, q, activation):
        a = activation[valid].float()
        counts[wrapper.key] += len(a)
        for block in a.split(cap["coactivation_chunk"]):
            matrices[wrapper.key].add_(block.T @ block)

    model.eval()
    controller.observe(observe)
    expected = {"encoder": 0, "decoder": 0}
    with fresh_output(path), terminal_log(path / "logs/capture.log"), torch.no_grad():
        for batch in make_loader(config, data, tokenizer, cap["coactivation_batch_size"]):
            x = move_batch(batch, device)
            expected["encoder"] += int(x["attention_mask"].sum())
            expected["decoder"] += int((x["labels"] != -100).sum())
            controller.teacher_batch(x)
            model(**x, use_cache=False)
        controller.observe(None)
        for key, value in matrices.items():
            array = value.cpu().numpy()
            if counts[key] != expected[controller.wrappers[key].stack] or counts[key] == 0:
                raise ValueError("E must include every valid token")
            if not np.isfinite(array).all() or not np.allclose(array, array.T, rtol=1e-6, atol=1e-5):
                raise ValueError("Non-finite/asymmetric coactivation matrix")
            np.savez(path / f"{key}.npz", coactivation_sum=array, valid_token_count=np.int64(counts[key]))
        complete(path, header)
        print(f"Captured full-validation dense E for {task}")
