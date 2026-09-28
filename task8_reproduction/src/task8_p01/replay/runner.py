from __future__ import annotations

from collections import defaultdict
import gzip
import json

import numpy as np

from task8_p01.assets.catalog import load_summaries
from task8_p01.common.config import MODES, TASKS, Snapshot, protocol_id, resolve_path
from task8_p01.common.io import checked_complete, complete, fresh_directory, write_csv, write_json
from task8_p01.compat.task6 import load_task6_config
from task8_p01.data.panel import freeze_panel, panel_members
from task8_p01.replay.core import replay_ffn
from task8_p01.substrate.model import attach, ffn_layers


def _root(config, snapshot, step):
    root = (
        resolve_path(config, config["execution"]["output_root"])
        / "runs/local_replay"
        / f"E_{snapshot.experts}" / f"k_{snapshot.k}"
    )
    return root / "init/step_0" if snapshot.role == "init" else root / f"best/seed_{snapshot.seed}/step_{step}"


def run_local_replay(config, snapshot, skip_complete=True):
    """Stream each natural hidden trace through all four local FFN cells.

    Three distinct forward passes provide Dense, M00, and M11 natural hidden
    sources. Within each callback, all four counterfactual FFN outputs see the
    exact same hidden tensor, so no preceding-layer drift contaminates a local
    comparison.
    """
    import torch

    if not isinstance(snapshot, Snapshot):
        snapshot = Snapshot(**snapshot)
    freeze_panel(config)
    task6_config = load_task6_config(config)
    from task6_phased.data.datasets import load_raw, make_loader, move_model_batch, tokenize
    from task6_phased.dense.model import load_t5
    from task6_phased.substrate.assets import inspect_assets
    paths, labels, c0, identity = inspect_assets(task6_config, snapshot.experts)
    learned, checkpoint = load_summaries(config, snapshot)
    root = _root(config, snapshot, checkpoint["step"])
    header = {
        "schema": 1, "protocol": protocol_id(config), "population": "diagnostic_128",
        **snapshot.to_dict(), "checkpoint_step": checkpoint["step"],
        "checkpoint_hash": checkpoint["state_sha256"],
        "trace_sources": ["dense", "M00", "M11"], "replay_modes": list(MODES),
        "teacher_forcing": "gold",
    }
    if root.exists():
        if not skip_complete:
            raise FileExistsError(root)
        checked_complete(root, header)
        return root

    reference_model, reference_tokenizer = load_t5(task6_config, paths["dense_best"], trainable=False)
    datasets = {}
    for task in TASKS:
        _, validation = load_raw(task6_config, task)
        encoded = tokenize(
            task6_config, task, validation, reference_tokenizer, "validation",
            identity["dense"]["model_sha256"],
        )
        members, _ = panel_members(config, task)
        datasets[task] = encoded.select(members)
    del reference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    values_by_group = defaultdict(list)
    with fresh_directory(root), gzip.open(root / "token_records.jsonl.gz", "wt", encoding="utf-8") as stream:
        current = {}

        def record(trace_source, layer, stack, hidden, valid, wi, wo, layer_labels):
            valid = valid.reshape(-1)
            selected_hidden = hidden.reshape(-1, hidden.shape[-1])[valid]
            replay = replay_ffn(
                selected_hidden, wi.weight, wo.weight, layer_labels,
                torch.as_tensor(c0[layer], device=selected_hidden.device),
                torch.as_tensor(learned[layer], device=selected_hidden.device),
                snapshot.k, config["routing"],
            )
            mask_2d = valid.reshape(current["batch_size"], -1)
            sample_rows, positions = torch.where(mask_2d)
            sample_ids = current["source_index"].index_select(0, sample_rows.cpu()).tolist()
            if stack == "encoder":
                content = current["content_mask"][sample_rows.cpu(), positions.cpu()].tolist()
                groups = ["encoder_content" if flag else "encoder_prefix" for flag in content]
            else:
                groups = ["decoder"] * len(sample_ids)
            for mode in MODES:
                result = replay[mode]
                weights = result["weights"].float()
                probabilities = weights / snapshot.k
                entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum(-1)
                neff = entropy.exp()
                errors = result["relative_l2_error"].float()
                for row in range(len(sample_ids)):
                    item = {
                        "phase": "P1", "E": snapshot.experts, "k": snapshot.k,
                        "seed": snapshot.seed, "checkpoint_role": snapshot.role,
                        "checkpoint_step": checkpoint["step"], "checkpoint_hash": checkpoint["state_sha256"],
                        "task": current["task"], "trace_source": trace_source,
                        "layer": layer, "stack": stack, "token_group": groups[row],
                        "sample_id": int(sample_ids[row]), "token_position": int(positions[row]),
                        "mode": mode, "relative_l2_error": float(errors[row]),
                        "max_weight": float(weights[row].max()), "n_eff_over_k": float(neff[row] / snapshot.k),
                    }
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    values_by_group[(trace_source, current["task"], layer, groups[row], mode)].append(
                        (item["relative_l2_error"], item["max_weight"], item["n_eff_over_k"])
                    )

        def run_source(trace_source):
            model, tokenizer = load_t5(task6_config, paths["dense_best"], trainable=False)
            controller = None
            handles = []
            if trace_source == "dense":
                for layer, stack, _, module in ffn_layers(model):
                    def hook(_module, args, _layer=layer, _stack=stack, _ffn=module):
                        valid = current["valid"][_stack]
                        record(trace_source, _layer, _stack, args[0], valid, _ffn.wi, _ffn.wo, torch.as_tensor(labels[_layer], device=args[0].device))
                    handles.append(module.register_forward_pre_hook(hook))
            else:
                controller = attach(config, model, labels, c0, learned, snapshot.k)
                controller.set_mode(trace_source)

                def observe(wrapper, hidden, valid, indices, weights, output):
                    del indices, weights, output
                    record(trace_source, wrapper.key, wrapper.stack, hidden, valid, wrapper.wi, wrapper.wo, wrapper.labels)

                controller.observe(observe)
            model.eval()
            with torch.no_grad():
                for task in TASKS:
                    loader = make_loader(
                        task6_config, datasets[task], tokenizer,
                        int(config["evaluation"]["teacher_batch_size"]),
                    )
                    for batch in loader:
                        model_batch = move_model_batch(batch, task6_config["execution"]["device"])
                        current.update({
                            "task": task, "batch_size": len(batch["source_index"]),
                            "source_index": batch["source_index"].cpu(),
                            "content_mask": batch["content_mask"].cpu(),
                            "valid": {
                                "encoder": model_batch["attention_mask"].bool(),
                                "decoder": model_batch["labels"] != -100,
                            },
                        })
                        if controller is not None:
                            controller.teacher_batch(model_batch)
                        model(**model_batch, use_cache=False)
            for handle in handles:
                handle.remove()
            if controller is not None:
                controller.observe(None)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for trace_source in ("dense", "M00", "M11"):
            run_source(trace_source)

        # gzip writes its footer on close. Finalize it before complete() hashes
        # the directory; the context manager's later close is harmless.
        stream.close()

        summary_rows = []
        for key, raw in sorted(values_by_group.items()):
            trace_source, task, layer, token_group, mode = key
            values = np.asarray(raw, dtype=float)
            row = {
                "phase": "P1", "E": snapshot.experts, "k": snapshot.k, "seed": snapshot.seed,
                "checkpoint_role": snapshot.role, "checkpoint_step": checkpoint["step"],
                "checkpoint_hash": checkpoint["state_sha256"], "task": task,
                "trace_source": trace_source, "layer": layer, "token_group": token_group,
                "mode": mode, "count": len(values),
            }
            for column, name in enumerate(("relative_l2_error", "max_weight", "n_eff_over_k")):
                row[f"{name}_mean"] = float(values[:, column].mean())
                row[f"{name}_median"] = float(np.median(values[:, column]))
                row[f"{name}_q25"] = float(np.quantile(values[:, column], .25))
                row[f"{name}_q75"] = float(np.quantile(values[:, column], .75))
            summary_rows.append(row)
        write_csv(root / "p1_local_replay.csv", summary_rows)
        write_json(root / "identity.json", header)
        complete(root, header)
    return root
