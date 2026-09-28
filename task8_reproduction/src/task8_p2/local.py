from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from task8_p01.common.config import digest
from task8_p01.common.io import write_csv, write_json
from task8_p2.artifacts import artifact_lock, atomic_artifact, read_npz, reuse, write_npz
from task8_p2.config import cell_fields, output_root
from task8_p2.numerics import Moments, chain, probabilities, support_metrics, weight_distance, weighting_metrics
from task8_p2.progress import Progress, log
from task8_p2.routing import router_arrays
from task8_p2.traces import GROUPS, align, build_trace, trace_path, verified_index


TABLES = {
    "support_natural": "p2_n2a_support_natural.csv",
    "support_shared": "p2_n2a_support_shared_hidden.csv",
    "weights": "p2_n2a_weight_summary.csv",
    "rank_n2a": "p2_n2a_weight_rank_profile.csv",
    "chain": "p2_n1_numeric_chain.csv",
    "rank_chain": "p2_n1_rank_profile.csv",
    "temperature": "p2_n3_local_temperature.csv",
    "rank_temperature": "p2_n3_rank_profile.csv",
}


def local_path(ctx, snapshot):
    scope = "smoke_local" if ctx.smoke else "local"
    return output_root(ctx.config) / "runs" / scope / ctx.identity()["protocol"][:16] / snapshot.path


class Collector:
    def __init__(self, base):
        self.base = base
        self.values, self.metadata, self.hashes = {}, {}, defaultdict(set)

    def add(self, table, meta, tokens, metrics=None, p=None, rank_table=None):
        n = len(tokens["sample_id"])
        group = tokens["token_group"]
        # Main view: token-pooled. Additional rows keep task, stack, layer and
        # token group separate; never average group means into the main view.
        scopes = [
            ({"task": "__all__", "layer_id": "__all__", "stack": "__all__", "token_group": "__all__"}, np.ones(n, bool)),
            ({"task": meta["task"], "layer_id": "__all__", "stack": "__all__", "token_group": "__all__"}, np.ones(n, bool)),
            ({"task": "__all__", "layer_id": "__all__", "stack": meta["stack"], "token_group": "__all__"}, np.ones(n, bool)),
            ({"task": "__all__", "layer_id": meta["layer_id"], "stack": meta["stack"], "token_group": "__all__"}, np.ones(n, bool)),
            ({"task": meta["task"], "layer_id": meta["layer_id"], "stack": meta["stack"], "token_group": "__all__"}, np.ones(n, bool)),
        ]
        for code in np.unique(group):
            scopes.append(({"task": meta["task"], "layer_id": meta["layer_id"], "stack": meta["stack"],
                            "token_group": GROUPS[int(code)]}, group == code))
            scopes.append(({"task": "__all__", "layer_id": "__all__", "stack": meta["stack"],
                            "token_group": GROUPS[int(code)]}, group == code))
        condition = {key: value for key, value in meta.items()
                     if key not in ("task", "layer_id", "stack", "hidden_hash", "token_group")}
        for scope, mask in scopes:
            row = {**self.base, **condition, **scope, "sample_id": "__all__", "token_position": "__all__"}
            key = (table, digest(row))
            if key not in self.values:
                self.values[key] = {}
                self.metadata[key] = row
            self.hashes[key].add(meta["hidden_hash"])
            if metrics is not None:
                for name, array in metrics.items():
                    self.values[key].setdefault(name, Moments()).add(np.asarray(array)[mask])
            if rank_table is not None:
                rkey = (rank_table, digest(row))
                if rkey not in self.values:
                    self.values[rkey] = {"rank": Moments()}
                    self.metadata[rkey] = row
                self.hashes[rkey].add(meta["hidden_hash"])
                self.values[rkey]["rank"].add(np.sort(p[mask], -1)[:, ::-1])

    def rows(self, table):
        result = []
        for key in sorted(self.values):
            if key[0] != table:
                continue
            row = {**self.metadata[key], "hidden_hash": digest(sorted(self.hashes[key])), "hidden_hash_scope": "aggregate_of_chunk_hashes"}
            values = self.values[key]
            if "rank" in values:
                mean, variance, std = values["rank"].values()
                for rank, (m, v, s) in enumerate(zip(mean, variance, std), 1):
                    result.append({**row, "task_scope": row["task"], "layer_scope": row["layer_id"],
                                   "token_group_scope": row["token_group"], "rank": rank,
                                   "mean_p": float(m), "variance_p": None if not np.isfinite(v) else float(v),
                                   "std_p": None if not np.isfinite(s) else float(s), "n_units": values["rank"].count})
            elif values:
                row["n_units"] = next(iter(values.values())).count
                for name, moment in values.items():
                    mean, variance, std = moment.values()
                    row[name] = float(mean)
                    row[f"{name}_unit_std"] = None if not np.isfinite(std) else float(std)
                result.append(row)
        return result


def pack_metrics(metrics):
    names = list(metrics)
    return np.asarray(names, dtype="U64"), np.column_stack([metrics[name] for name in names])


def temperatures(selected):
    base = probabilities(selected)
    uniform = probabilities(selected, "uniform")
    previous_variance = None
    result = {}
    for multiplier in (1, 2, 4, 8):
        p = probabilities(selected, multiplier)
        metrics = weighting_metrics(selected/multiplier, p)
        metrics["weight_l1_to_m1"] = weight_distance(p, base)["l1"]
        metrics["weight_l1_to_uniform"] = weight_distance(p, uniform)["l1"]
        if previous_variance is not None and np.any(metrics["weight_variance"] > previous_variance+1e-9):
            raise ValueError("Temperature flattening failed per-unit monotonicity check")
        previous_variance = metrics["weight_variance"]
        result[multiplier] = (metrics, p)
    return result


def analyze_snapshot(ctx, snapshot):
    learned, checkpoint = ctx.snapshot(snapshot)
    _, _, c0, assets = ctx.assets(snapshot.experts)
    dense_path = trace_path(ctx, "dense")
    r2_path = trace_path(ctx, "natural_R2", experts=snapshot.experts, k=snapshot.k)
    # Formal launcher prepares these once. Individual jobs may safely construct
    # a missing shared trace under its cross-process advisory lock.
    if not dense_path.exists():
        build_trace(ctx, "dense")
    if not r2_path.exists():
        build_trace(ctx, "natural_R2", experts=snapshot.experts, k=snapshot.k)
    r4_path = build_trace(ctx, "natural_R4d", snapshot=snapshot)
    dense_index, r2_index, r4_index = [verified_index(ctx, p) for p in (dense_path, r2_path, r4_path)]
    indices = [{row["key"]: row for row in index["chunks"]} for index in (dense_index, r2_index, r4_index)]
    if not (set(indices[0]) == set(indices[1]) == set(indices[2])):
        raise ValueError("Dense/R2/R4d traces have different task/batch/layer keys")
    identity = {**ctx.identity(), "artifact": "p2_local", **cell_fields(snapshot.experts, snapshot.k),
                "seed": snapshot.seed, "checkpoint_role": "best", "checkpoint_step": checkpoint["step"],
                "checkpoint_hash": checkpoint["state_sha256"], "checkpoint_summary_hash": checkpoint["summary_sha256"],
                "dense_hash": assets["dense"]["model_sha256"], "split_hash": assets["split"]["files"],
                "trace_paths": {"dense": str(dense_path), "natural_R2": str(r2_path), "natural_R4d": str(r4_path)}}
    base = {key: value for key, value in identity.items()
            if key not in ("data_definition", "split_hash", "trace_paths", "artifact", "schema")}
    base.update({"tie_atol": ctx.config["p2"]["tie_atol"], "tie_rtol": ctx.config["p2"]["tie_rtol"],
                 "tie_break": ctx.config["routing"]["tie_break"], "gap_delta": ctx.config["p2"]["gap_delta"],
                 "router_dimension": 512, "global_logit_scale": math.sqrt(512)/ctx.config["routing"]["temperature"],
                 "rms_epsilon": ctx.config["routing"]["rms_epsilon"]})
    path = local_path(ctx, snapshot)
    with artifact_lock(path):
        if reuse(path, identity):
            return path
        with atomic_artifact(path, identity) as staging:
            collector = Collector(base)
            raw_index = []
            progress = Progress(f"LOCAL E{snapshot.experts}/k{snapshot.k}/seed{snapshot.seed}", len(indices[0]), ctx.config["p2"]["progress_interval_seconds"])
            for batch_number, key in enumerate(sorted(indices[0])):
                rows = [index[key] for index in indices]
                dense, r2, r4 = [read_npz(root / row["file"]) for root, row in zip((dense_path, r2_path, r4_path), rows)]
                align(dense, r2)
                align(r2, r4)
                width = int(ctx.config["p2"]["router_chunk_size"])
                for start in range(0, len(dense["sample_id"]), width):
                    stop = min(start+width, len(dense["sample_id"]))
                    chunk = {name: value[start:stop] for name, value in dense.items()}
                    meta = {"task": rows[0]["task"], "layer_id": rows[0]["layer_id"], "stack": rows[0]["stack"]}
                    layer = meta["layer_id"]
                    route = router_arrays(chunk["hidden"], c0[layer], learned[layer], snapshot.k, ctx.config["routing"], ctx.device)
                    p2_cfg = ctx.config["p2"]
                    support_args = {"atol": p2_cfg["tie_atol"], "rtol": p2_cfg["tie_rtol"], "delta": p2_cfg["gap_delta"]}
                    shared_support = support_metrics(route["ids_R2"], route["ids_R4d"], route["scores_R2"], route["z_actual"], **support_args)
                    natural_support = support_metrics(r2["topk_ids"][start:stop], r4["topk_ids"][start:stop],
                                                      r2["all_scores"][start:stop], r4["all_scores"][start:stop], **support_args)
                    shared_meta = {**meta, "forward_mode": "shared_hidden", "hidden_source": "dense",
                                   "hidden_hash": rows[0]["hidden_hash"], "support_source": "I4_vs_I2"}
                    natural_meta = {**meta, "forward_mode": "natural", "hidden_source": "natural_R2_vs_R4d",
                                    "hidden_hash": digest([rows[1]["hidden_hash"], rows[2]["hidden_hash"]]), "support_source": "I4_vs_I2"}
                    collector.add("support_shared", shared_meta, chunk, shared_support)
                    collector.add("support_natural", natural_meta, chunk, natural_support)
                    raw = {name: chunk[name] for name in ("sample_id", "token_position", "token_group")}
                    raw.update(route)
                    raw["natural_ids_R2"] = r2["topk_ids"][start:stop]
                    raw["natural_ids_R4d"] = r4["topk_ids"][start:stop]
                    raw["support_metric_names"], raw["support_shared_values"] = pack_metrics(shared_support)
                    _, raw["support_natural_values"] = pack_metrics(natural_support)
                    for mode, selected, p, hhash, hsource in (
                        ("shared_hidden", route["selected_scores"], route["p"], rows[0]["hidden_hash"], "dense"),
                        ("natural", r4["selected_scores"][start:stop], r4["p"][start:stop], rows[2]["hidden_hash"], "natural_R4d"),
                    ):
                        metrics = weighting_metrics(selected, p)
                        wmeta = {**meta, "forward_mode": mode, "hidden_source": hsource, "hidden_hash": hhash, "support_source": "I4"}
                        collector.add("weights", wmeta, chunk, metrics, p, "rank_n2a")
                        raw["weight_metric_names"], raw[f"weight_{mode}_values"] = pack_metrics(metrics)
                        tvalues = temperatures(selected)
                        if not np.allclose(tvalues[1][1], p, atol=2e-6, rtol=2e-6):
                            raise ValueError("m=1 local weights differ from the original endpoint")
                        for multiplier, (tmetrics, tp) in tvalues.items():
                            tmeta = {**wmeta, "multiplier": multiplier,
                                     "forward_mode": "shared_hidden" if mode == "shared_hidden" else "natural_R4d_replay"}
                            collector.add("temperature", tmeta, chunk, tmetrics, tp, "rank_temperature")
                            raw["temperature_metric_names"], raw[f"temperature_{mode}_m{multiplier}_values"] = pack_metrics(tmetrics)
                            raw[f"temperature_{mode}_m{multiplier}_p"] = tp
                    for support_name, ids in (("I4", route["ids_R4d"]), ("I2", route["ids_R2"])):
                        nvalues = chain(route["q_St"], route["z_actual"], route["eta_x"], route["eta_St"], ids,
                                        math.sqrt(chunk["hidden"].shape[-1])/ctx.config["routing"]["temperature"])
                        previous_variance = None
                        for state, (nmetrics, np_) in nvalues.items():
                            nmetrics["delta_weight_variance_to_previous"] = (
                                np.zeros(len(np_)) if previous_variance is None else nmetrics["weight_variance"]-previous_variance)
                            previous_variance = nmetrics["weight_variance"]
                            nmeta = {**shared_meta, "support_source": support_name, "n_state": state}
                            collector.add("chain", nmeta, chunk, nmetrics, np_, "rank_chain")
                            raw["chain_metric_names"], raw[f"chain_{support_name}_{state}_values"] = pack_metrics(nmetrics)
                            raw[f"chain_{support_name}_{state}_p"] = np_
                    raw_name = f"raw/{meta['task']}/{layer}/batch_{key.split(':')[1]}_offset_{start:06d}.npz"
                    write_npz(staging/raw_name, **raw)
                    raw_index.append({**base, **meta, "file": raw_name, "offset": start, "n_units": stop-start,
                                      "forward_mode": "natural_and_shared", "hidden_source": "dense_plus_natural_R2_R4d",
                                      "hidden_hash": digest([r["hidden_hash"] for r in rows]), "support_source": "I2_and_I4",
                                      "dense_hidden_file": str(dense_path/rows[0]["file"]), "dense_hidden_hash": rows[0]["hidden_hash"],
                                      "r2_hidden_file": str(r2_path/rows[1]["file"]), "r2_hidden_hash": rows[1]["hidden_hash"],
                                      "r4d_hidden_file": str(r4_path/rows[2]["file"]), "r4d_hidden_hash": rows[2]["hidden_hash"]})
                progress.update(batch_number+1)
            for table, filename in TABLES.items():
                write_csv(staging/filename, collector.rows(table))
            write_csv(staging/"p2_local_units_index.csv", raw_index)
            write_json(staging/"checks.json", {"passed": True, "natural_alignment": True,
                       "numeric_chain_matches_actual": True, "m1_matches_endpoint": True,
                       "temperature_variance_monotone_per_unit": True,
                       "fixed_support_across_numeric_and_temperature_states": True,
                       "chunks": len(raw_index), "n_units_per_forward_mode": sum(r["n_units"] for r in raw_index),
                       "raw_schema": "NPZ arrays: aligned IDs; *_metric_names define columns of *_values; *_p holds unit×k probabilities"})
    log(f"LOCAL_COMPLETE {snapshot} {path}")
    return path
