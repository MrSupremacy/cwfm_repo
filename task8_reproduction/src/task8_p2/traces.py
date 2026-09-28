from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from task8_p01.common.config import TASKS, digest
from task8_p01.common.io import read_json, write_json
from task8_p2.artifacts import artifact_lock, atomic_artifact, read_npz, reuse, write_npz
from task8_p2.config import cell_fields
from task8_p2.progress import ProgressLoader, log


GROUPS = ("encoder_prefix", "encoder_content", "decoder")


def trace_path(ctx, source, snapshot=None, experts=None, k=None):
    root = ctx.namespace() / "traces" / source
    if source == "dense":
        return root / "shared"
    if source == "natural_R2":
        return root / f"E_{experts}" / f"k_{k}" / "static"
    return root / snapshot.path


def verified_index(ctx, path):
    path = Path(path)
    if path not in ctx.verified:
        if not reuse(path):
            raise FileNotFoundError(f"Required trace is missing: {path}; run prepare/local first")
        ctx.verified.add(path)
    return read_json(path / "index.json")


def build_trace(ctx, source, snapshot=None, experts=None, k=None):
    import torch
    from task6_phased.data.datasets import make_loader, move_model_batch
    from task8_p01.routing.interventions import l2_logits, rms_logits
    from task8_p01.substrate.model import attach, ffn_layers

    if source not in ("dense", "natural_R2", "natural_R4d"):
        raise ValueError(source)
    if snapshot is not None:
        experts, k = snapshot.experts, snapshot.k
    paths, labels, c0, assets = ctx.assets(experts or 64)
    learned, checkpoint = (ctx.snapshot(snapshot) if source == "natural_R4d" else (c0, None))
    identity = {**ctx.identity(), "artifact": "hidden_trace", "hidden_source": source,
                "teacher_batch_size": int(ctx.config["p2"]["teacher_batch_size"]),
                "cache_dtype": "float32", "dense_hash": assets["dense"]["model_sha256"]}
    if source != "dense":
        identity.update({**cell_fields(experts, k), "split_hash": assets["split"]["files"],
                         "seed": snapshot.seed if snapshot else None,
                         "checkpoint_role": "best" if snapshot else "static",
                         "checkpoint_step": checkpoint["step"] if checkpoint else 0,
                         "checkpoint_hash": checkpoint["state_sha256"] if checkpoint else None,
                         "checkpoint_summary_hash": checkpoint["summary_sha256"] if checkpoint else None})
    path = trace_path(ctx, source, snapshot, experts, k)
    with artifact_lock(path):
        if reuse(path, identity):
            ctx.verified.add(path)
            return path
        with atomic_artifact(path, identity) as staging:
            model, tokenizer = ctx.load_model(experts or 64)
            datasets = ctx.datasets(tokenizer)
            current, rows, handles = {}, [], []
            controller = None

            def record(layer, stack, hidden, valid, indices=None, weights=None):
                valid = valid.reshape(-1)
                h = hidden.reshape(-1, hidden.shape[-1])[valid].float()
                valid_2d = valid.reshape(current["batch_size"], -1)
                row, position = torch.where(valid_2d)
                sample_ids = current["source_index"].index_select(0, row.cpu()).numpy()
                if stack == "encoder":
                    group = current["content_mask"][row.cpu(), position.cpu()].numpy().astype(np.uint8)
                else:
                    group = np.full(len(row), 2, np.uint8)
                values = {"hidden": h.detach().cpu().numpy(), "sample_id": sample_ids,
                          "token_position": position.cpu().numpy(), "token_group": group}
                if indices is not None:
                    valid_ids = indices[valid]
                    values["topk_ids"] = valid_ids.detach().cpu().numpy()
                    values["p"] = (weights[valid].float()/k).detach().cpu().numpy()
                    if source == "natural_R2":
                        scores = l2_logits(h, torch.as_tensor(c0[layer], device=h.device), ctx.config["routing"]["l2_epsilon"])
                    else:
                        scores = rms_logits(h, torch.as_tensor(learned[layer], device=h.device),
                                            ctx.config["routing"]["rms_epsilon"], ctx.config["routing"]["temperature"])
                    values["all_scores"] = scores.detach().cpu().numpy()
                    values["selected_scores"] = scores.gather(-1, valid_ids).detach().cpu().numpy()
                if not np.isfinite(values["hidden"]).all():
                    raise ValueError(f"Nonfinite hidden: {source}/{layer}")
                filename = f"{current['task']}/batch_{current['batch_index']:04d}/{layer}.npz"
                write_npz(staging / filename, **values)
                rows.append({"key": f"{current['task']}:{current['batch_index']}:{layer}",
                             "file": filename, "task": current["task"], "layer_id": layer,
                             "stack": stack, "n_units": len(h),
                             "hidden_hash": hashlib.sha256(values["hidden"].tobytes()).hexdigest()})

            if source == "dense":
                for layer, stack, _, module in ffn_layers(model):
                    def hook(_module, args, _layer=layer, _stack=stack):
                        record(_layer, _stack, args[0], current["valid"][_stack])
                    handles.append(module.register_forward_pre_hook(hook))
            else:
                controller = attach(ctx.legacy, model, labels, c0, learned, k)
                controller.set_mode("M00" if source == "natural_R2" else "M11")
                def observe(wrapper, hidden, valid, indices, weights, output):
                    record(wrapper.key, wrapper.stack, hidden, valid, indices, weights)
                controller.observe(observe)
            try:
                model.eval()
                with torch.no_grad():
                    for task in TASKS:
                        loader = make_loader(ctx.task6, datasets[task], tokenizer, int(ctx.config["p2"]["teacher_batch_size"]))
                        for batch_index, batch in enumerate(ProgressLoader(loader, f"TRACE {source} {task}", ctx.config["p2"]["progress_interval_seconds"])):
                            model_batch = move_model_batch(batch, ctx.device)
                            current.update({"task": task, "batch_index": batch_index, "batch_size": len(batch["source_index"]),
                                            "source_index": batch["source_index"].cpu(), "content_mask": batch["content_mask"].cpu(),
                                            "valid": {"encoder": model_batch["attention_mask"].bool(), "decoder": model_batch["labels"] != -100}})
                            if controller is not None:
                                controller.teacher_batch(model_batch)
                            model(**model_batch, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if len({row["key"] for row in rows}) != len(rows):
                raise ValueError("Trace has duplicate task/batch/layer keys")
            write_json(staging / "index.json", {"identity": identity, "chunks": rows,
                                               "n_units": sum(row["n_units"] for row in rows), "token_group_codes": dict(enumerate(GROUPS))})
        ctx.verified.add(path)
    log(f"TRACE_COMPLETE {source} {path}")
    return path


def align(left, right):
    for key in ("sample_id", "token_position", "token_group"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Natural traces lost task/sample/token/layer alignment at {key}")


def prepare(ctx):
    from task8_p01.common.config import cells
    build_trace(ctx, "dense")
    for cell in cells():
        build_trace(ctx, "natural_R2", experts=cell.experts, k=cell.k)
