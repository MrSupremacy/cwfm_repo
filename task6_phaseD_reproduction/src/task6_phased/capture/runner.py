from __future__ import annotations

import gzip
import json

import numpy as np

from task6_phased.capture.storage import ProbeWriter, write_parquet
from task6_phased.common.config import R4_FAMILY, TASKS, protocol_id, run_path
from task6_phased.common.context import load_context
from task6_phased.common.io import checked_complete, complete, fresh_output
from task6_phased.data.datasets import make_loader, move_model_batch
from task6_phased.dense.evaluation import evaluate_model
from task6_phased.training.checkpoints import load_for_capture, states


def capture_path(config, condition, run_id, state, kind, task=None):
    path = run_path(config, f"capture/{kind}", condition, run_id) / state["name"]
    return path if task is None else path / task


def _header(config, condition, state, inputs, kind, task=None, with_q=False):
    return {
        "schema": 2, "protocol": protocol_id(config), "condition": condition.to_dict(),
        "state": state, "input_header": inputs, "kind": kind, "task": task, "with_q": with_q,
    }


def _reuse(path, header, skip):
    if skip and path.exists():
        checked_complete(path, header)
        return True
    return False


def capture_predictions(config, condition, run_id, skip_complete=False):
    model, tokenizer, controller, _, inputs = load_context(
        config, condition, run_id, populations=("validation",)
    )
    for state in states(config, condition, run_id):
        path = capture_path(config, condition, run_id, state, "A")
        header = _header(config, condition, state, inputs, "A")
        if _reuse(path, header, skip_complete):
            continue
        load_for_capture(config, condition, run_id, state, controller, inputs)
        controller.generation()
        with fresh_output(path):
            result = evaluate_model(config, model, tokenizer)
            for task in TASKS:
                with gzip.open(path / f"{task}.json.gz", "wt", encoding="utf-8") as stream:
                    json.dump(result[task], stream, ensure_ascii=False, allow_nan=False)
            (path / "summary.json").write_text(
                json.dumps(result["summary"], indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            complete(path, header)


def _special_mask(ids, special_ids):
    import torch
    result = torch.zeros_like(ids, dtype=torch.bool)
    for value in special_ids:
        result |= ids == int(value)
    return result


def capture_diagnostics(config, condition, run_id, skip_complete=False):
    import torch
    from task6_phased.metrics.performance.pipeline import best_state
    model, tokenizer, controller, datasets, inputs = load_context(
        config, condition, run_id, populations=("validation", "probe")
    )
    selected_best = best_state(config, condition, run_id)["name"] if condition.trainable else "static"
    final_name = f"step_{config['training']['total_optimizer_steps']}" if condition.trainable else "static"
    special_ids = tokenizer.all_special_ids
    for state in states(config, condition, run_id):
        load_for_capture(config, condition, run_id, state, controller, inputs)
        model.eval()
        modes = ["probe"]
        if state["name"] in {selected_best, final_name, "static"}:
            modes.insert(0, "B")
        # Phase D captures local activation sums for every routed state and arm;
        # overlap/coverage trajectories must never be reconstructed from another arm's hidden states.
        with_q = True
        for task in TASKS:
            for mode in modes:
                path = capture_path(config, condition, run_id, state, mode, task)
                header = _header(config, condition, state, inputs, mode, task, with_q and mode == "probe")
                if _reuse(path, header, skip_complete):
                    continue
                dataset = datasets[task]["probe" if mode == "probe" else "validation"]
                loader = make_loader(
                    config, dataset, tokenizer,
                    config["capture"]["probe_batch_size" if mode == "probe" else "teacher_batch_size"],
                )
                counts = {
                    key: np.zeros(condition.experts, dtype=np.int64)
                    for key in controller.wrappers
                }
                tokens = dict.fromkeys(controller.wrappers, 0)
                with fresh_output(path), torch.no_grad():
                    writers = {
                        key: ProbeWriter(
                            path / key, key, None, condition.k, condition.experts, with_q,
                            config["capture"]["shard_rows"], config["capture"]["compression_level"],
                        )
                        for key in controller.wrappers
                    } if mode == "probe" else {}

                    def observe(wrapper, shape, valid, selected, q):
                        valid_selected = selected[valid].detach().cpu().numpy()
                        if mode == "B":
                            counts[wrapper.key] += np.bincount(
                                valid_selected.reshape(-1), minlength=condition.experts
                            )
                            tokens[wrapper.key] += len(valid_selected)
                            return
                        rows, positions = torch.where(valid.reshape(shape))
                        ids = token_ids[wrapper.stack]
                        row_cpu, position_cpu = rows.cpu(), positions.cpu()
                        selected_ids = ids[row_cpu, position_cpu]
                        special = _special_mask(selected_ids, special_ids)
                        if wrapper.stack == "encoder":
                            content = content_masks[row_cpu, position_cpu] & (~special)
                        else:
                            content = ~special
                        keys = np.column_stack((
                            sample_ids[row_cpu.numpy()], position_cpu.numpy()
                        ))
                        writers[wrapper.key].add(
                            keys, valid_selected, None if not with_q else q[valid].cpu().numpy(),
                            domain=task, stack=wrapper.stack,
                            is_content=content.numpy(), is_special=special.numpy(),
                        )

                    controller.observe(observe, with_q=mode == "probe" and with_q)
                    expected_tokens = {"encoder": 0, "decoder": 0}
                    for batch in loader:
                        sample_ids = batch["source_index"].numpy()
                        content_masks = batch["content_mask"]
                        values = move_model_batch(batch, config["execution"]["device"])
                        controller.teacher_batch(values)
                        token_ids = {
                            "encoder": values["input_ids"].detach().cpu(),
                            "decoder": values["labels"].detach().cpu(),
                        }
                        expected_tokens["encoder"] += int(values["attention_mask"].sum())
                        expected_tokens["decoder"] += int((values["labels"] != -100).sum())
                        model(**values, use_cache=False)
                    controller.observe(None)
                    if mode == "B":
                        for key, wrapper in controller.wrappers.items():
                            if tokens[key] != expected_tokens[wrapper.stack] or counts[key].sum() != tokens[key] * condition.k:
                                raise ValueError(f"{task}/{key}: Tk conservation failed")
                        keys = list(counts)
                        write_parquet(
                            path / "loads.parquet",
                            {"layer_id": keys, "valid_token_count": [tokens[key] for key in keys],
                             "assignment_counts": np.stack([counts[key] for key in keys])},
                            "B", condition.experts, condition.k,
                            config["capture"]["compression_level"],
                        )
                    else:
                        for writer in writers.values():
                            writer.finish()
                    complete(path, header)
