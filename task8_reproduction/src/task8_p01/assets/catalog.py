from __future__ import annotations

import hashlib
import os
from pathlib import Path

from task8_p01.common.config import (
    BUDGETS, EXPERTS, Snapshot, implementation_id, resolve_path, snapshots,
)
from task8_p01.common.io import path_identity, read_json, sha256, write_csv, write_json
from task8_p01.compat.task6 import load_task6_config, task6_source_identity


CATALOG_SCHEMA = 2


def _key(snapshot):
    if snapshot.role == "init":
        return f"E{snapshot.experts}_k{snapshot.k}_init"
    return f"E{snapshot.experts}_k{snapshot.k}_seed{snapshot.seed}_best"


def read_catalog(config):
    path = resolve_path(config, config["assets"]["r4d_catalog"])
    value = read_json(path)
    if value.get("schema") != CATALOG_SCHEMA or value.get("checkpoint_roles") != ["init", "best"]:
        raise ValueError(f"Unsupported init/best checkpoint catalog: {path}")
    expected = {_key(snapshot) for snapshot in snapshots()}
    entries = value.get("entries", {})
    if set(entries) != expected:
        missing, extra = expected - set(entries), set(entries) - expected
        raise ValueError(f"Catalog is not D9 init + D9x3 best complete; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def resolve_snapshot(config, snapshot):
    if not isinstance(snapshot, Snapshot):
        snapshot = Snapshot(**snapshot)
    entry = read_catalog(config)["entries"][_key(snapshot)]
    path = Path(entry["checkpoint_path"]).expanduser().resolve(strict=True)
    if path.name != f"step_{int(entry['step'])}":
        raise ValueError(f"Catalog step/path mismatch: {path}")
    for required in ("state.pt", "meta.json", "complete.json"):
        if not (path / required).is_file():
            raise FileNotFoundError(path / required)
    if sha256(path / "state.pt") != entry["state_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    return path, entry


def _summary_tensors(path, experts):
    import torch

    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    routers = state.get("routers")
    expected_layers = {
        *(f"encoder_layer_{index:02d}" for index in range(6)),
        *(f"decoder_layer_{index:02d}" for index in range(6)),
    }
    if not isinstance(routers, dict) or set(routers) != expected_layers:
        raise ValueError(f"R4d checkpoint does not contain exactly 12 router layers: {path}")
    result = {}
    for layer, router_state in routers.items():
        if set(router_state) != {"summary"}:
            raise ValueError(f"{layer}: expected only the learned R4d summary tensor")
        summary = router_state["summary"].detach().float().cpu().contiguous()
        if tuple(summary.shape) != (int(experts), 512) or not torch.isfinite(summary).all():
            raise ValueError(f"{layer}: invalid summary shape/value")
        result[layer] = summary
    return result


def _summary_identity(summaries):
    value = hashlib.sha256()
    for layer, tensor in sorted(summaries.items()):
        value.update(layer.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.numpy().tobytes(order="C"))
    return value.hexdigest()


def load_summaries(config, snapshot):
    path, entry = resolve_snapshot(config, snapshot)
    result = _summary_tensors(path, snapshot.experts)
    if _summary_identity(result) != entry["summary_sha256"]:
        raise ValueError(f"Router-summary hash mismatch: {path}")
    return result, {**entry, "resolved_checkpoint_path": str(path)}


def inventory(config, output=None):
    """Verify all referenced assets and emit a machine-readable P0 inventory."""
    task6_config = load_task6_config(config)
    from task6_phased.substrate.assets import inspect_assets

    source_root = Path(__file__).resolve().parents[3] / "src/task8_p01"
    rows = [{
        "asset_type": "task8_source", "identity": implementation_id(),
        "declared_path": str(source_root), "resolved_path": str(source_root),
        "experts": "", "k": "", "seed": "", "role": "", "step": "", "status": "ok",
    }]
    source = task6_source_identity(config)
    rows.append({
        "asset_type": "task6_source", "identity": source["sha256"],
        "declared_path": source["root"], "resolved_path": source["root"],
        "experts": "", "k": "", "seed": "", "role": "", "step": "", "status": "ok",
    })
    seen_dense = False
    for experts in EXPERTS:
        paths, _, _, identity = inspect_assets(task6_config, experts)
        if not seen_dense:
            dense = path_identity(paths["dense_best"])
            rows.append({
                "asset_type": "dense_best", "identity": identity["dense"]["model_sha256"],
                "declared_path": dense["declared_path"], "resolved_path": dense["resolved_path"],
                "experts": "", "k": "", "seed": "", "role": "", "step": 2376, "status": "ok",
            })
            seen_dense = True
        split = path_identity(paths["expert_split"])
        rows.append({
            "asset_type": "expert_split", "identity": split["sha256"],
            "declared_path": split["declared_path"], "resolved_path": split["resolved_path"],
            "experts": experts, "k": "", "seed": 1, "role": "", "step": "", "status": "ok",
        })
    catalog = read_catalog(config)
    for snapshot in snapshots():
        path, entry = resolve_snapshot(config, snapshot)
        load_summaries(config, snapshot)
        rows.append({
            "asset_type": f"r4d_{snapshot.role}", "identity": entry["summary_sha256"],
            "declared_path": entry.get("link_path", entry["checkpoint_path"]),
            "resolved_path": str(path), "experts": snapshot.experts, "k": snapshot.k,
            "seed": "" if snapshot.seed is None else snapshot.seed, "role": snapshot.role,
            "step": entry["step"], "status": "ok",
        })
    if output:
        write_csv(output, rows)
    return {"schema": CATALOG_SCHEMA, "checkpoint_roles": ["init", "best"], "catalog": catalog, "rows": rows}


def _link_once(source, link, create_links):
    if not create_links:
        return link if link.exists() else source
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve(strict=True) != source.resolve(strict=True):
            raise FileExistsError(f"Refusing to replace mismatched link: {link}")
    else:
        os.symlink(source.resolve(strict=True), link, target_is_directory=True)
    return link


def write_catalog(path, entries, source_run_id="routed02"):
    payload = {
        "schema": CATALOG_SCHEMA, "checkpoint_roles": ["init", "best"],
        "source_run_id": source_run_id, "entries": entries,
    }
    write_json(path, payload)
    return payload


def build_catalog(config, *, create_links=False):
    """Resolve one seed-deduplicated init and three registered best states per D9 cell."""
    import torch

    task6_config = load_task6_config(config)
    from task6_phased.common.config import Condition, run_path
    from task6_phased.common.io import checked_complete as task6_checked_complete
    from task6_phased.substrate.assets import inspect_assets

    source_run_id = config["protocol"]["source_run_id"]
    init_root = resolve_path(config, config["assets"]["r4d_init_root"])
    best_root = resolve_path(config, config["assets"]["r4d_best_root"])
    entries = {}
    for experts in EXPERTS:
        _, _, centroids, _ = inspect_assets(task6_config, experts)
        for k in BUDGETS[experts]:
            init_candidates = []
            for seed in (0, 1, 2):
                condition = Condition(experts, "R4d", k, seed)
                source = run_path(task6_config, "train", condition, source_run_id) / "checkpoints/step_0"
                meta = task6_checked_complete(source)
                if meta["step"] != 0 or meta["condition"] != condition.to_dict():
                    raise ValueError(f"Init checkpoint metadata mismatch: {source}")
                summaries = _summary_tensors(source, experts)
                init_candidates.append((seed, source, summaries, _summary_identity(summaries)))
            hashes = {item[3] for item in init_candidates}
            if len(hashes) != 1:
                raise ValueError(f"E={experts},k={k}: step0 summaries differ across seeds")
            canonical_seed, source, summaries, summary_hash = init_candidates[0]
            for layer, summary in summaries.items():
                expected = torch.as_tensor(centroids[layer]).float()
                if not torch.equal(summary, expected):
                    raise ValueError(f"E={experts},k={k},{layer}: init summary is not bitwise C0")
            link = init_root / f"E_{experts}" / f"k_{k}"
            checkpoint_path = _link_once(source, link, create_links)
            init_snapshot = Snapshot(experts, k, None, "init")
            entries[_key(init_snapshot)] = {
                "experts": experts, "k": k, "seed": None, "role": "init",
                "canonical_source_seed": canonical_seed, "verified_source_seeds": [0, 1, 2],
                "init_equals_c0": True, "step": 0, "state_name": "step_0",
                "checkpoint_path": str(checkpoint_path.absolute()),
                "source_checkpoint_path": str(source.resolve(strict=True)), "link_path": str(link.absolute()),
                "state_sha256": sha256(source / "state.pt"), "summary_sha256": summary_hash,
                "source_state_sha256_by_seed": {
                    str(seed): sha256(candidate / "state.pt") for seed, candidate, _, _ in init_candidates
                },
            }

            for seed in (0, 1, 2):
                condition = Condition(experts, "R4d", k, seed)
                selection_path = run_path(task6_config, "metrics/performance", condition, source_run_id) / "selection.json"
                selection = read_json(selection_path)
                if selection.get("condition") != condition.to_dict():
                    raise ValueError(f"Selection condition mismatch: {selection_path}")
                state = selection["state"]
                source = run_path(task6_config, "train", condition, source_run_id) / "checkpoints" / state["name"]
                meta = task6_checked_complete(source)
                if meta["step"] != state["step"] or meta["condition"] != condition.to_dict():
                    raise ValueError(f"Selected checkpoint metadata mismatch: {source}")
                summaries = _summary_tensors(source, experts)
                link = best_root / f"E_{experts}" / f"k_{k}" / f"seed_{seed}"
                checkpoint_path = _link_once(source, link, create_links)
                chosen = next(
                    (item for item in selection.get("candidates", []) if item["state"]["name"] == state["name"]),
                    None,
                )
                snapshot = Snapshot(experts, k, seed, "best")
                entries[_key(snapshot)] = {
                    "experts": experts, "k": k, "seed": seed, "role": "best",
                    "step": int(state["step"]), "state_name": state["name"],
                    "checkpoint_path": str(checkpoint_path.absolute()),
                    "source_checkpoint_path": str(source.resolve(strict=True)), "link_path": str(link.absolute()),
                    "state_sha256": sha256(source / "state.pt"),
                    "summary_sha256": _summary_identity(summaries),
                    "selection_path": str(selection_path.resolve(strict=True)),
                    "selection_sha256": sha256(selection_path),
                    "old_macro": None if chosen is None else chosen.get("macro"),
                    "old_worst_domain": None if chosen is None else chosen.get("worst_domain"),
                }
    path = resolve_path(config, config["assets"]["r4d_catalog"])
    return write_catalog(path, entries, source_run_id)
