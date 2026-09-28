from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import re

from task8_p01.common.config import (
    BUDGETS, SEEDS, Snapshot, cells, digest, implementation_id as p1_code_id,
    load_config as load_p1_config, repository_root, resolve_path,
)


def load_config(suite=None, local=None):
    config = load_p1_config(suite or repository_root() / "configs/suites/p2_n_forward.yaml", local)
    p2 = config["p2"]
    if p2["experiments"] != ["N2-a", "N1", "N3"] or p2["checkpoint_role"] != "best":
        raise ValueError("P2 scope is frozen to N2-a/N1/N3 and best checkpoints")
    if p2["multipliers"] != [1, 2, 4, 8] or p2["auto_select_temperature"] is not False:
        raise ValueError("P2 requires m=1/2/4/8; no automatic temperature selection")
    if p2["cache_dtype"] != "float32":
        raise ValueError("Hidden caches must retain float32 values")
    for key in ("router_chunk_size", "teacher_batch_size", "smoke_samples_per_task"):
        if int(p2[key]) <= 0:
            raise ValueError(f"p2.{key} must be positive")
    for key in ("gap_delta", "progress_interval_seconds", "heartbeat_seconds"):
        if float(p2[key]) <= 0:
            raise ValueError(f"p2.{key} must be positive")
    for key in ("tie_atol", "tie_rtol"):
        if float(p2[key]) < 0:
            raise ValueError(f"p2.{key} must be nonnegative")
    evaluation = config["evaluation"]
    expected_scoring = {"candidate_append_eos": True, "candidate_length_normalized": True,
                        "candidate_tie_break": "lower_label_id", "store_candidate_scores": True}
    if any(evaluation.get(key) != value for key, value in expected_scoring.items()):
        raise ValueError("P2 must preserve the P1 candidate-label scoring contract")
    root = output_root(config)
    p1 = p1_root(config)
    if root == p1 or p1 in root.parents or root in p1.parents:
        raise ValueError("P2 output must be separate from, not nested in, P0/P1 output")
    return config


def legacy_config(config):
    value = deepcopy(config)
    value.pop("p2", None)
    return value


def output_root(config):
    return resolve_path(config, config["p2"]["output_root"])


def p1_root(config):
    return resolve_path(config, config["p2"]["p1_output_root"])


def validate_run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) or value in (".", ".."):
        raise ValueError("run-id must be a simple identifier, not a path")
    return value


def result_root(config, run_id):
    return output_root(config) / "results" / validate_run_id(run_id)


@lru_cache(maxsize=1)
def code_id():
    root = repository_root() / "src/task8_p2"
    return digest({
        "p1_library": p1_code_id(),
        "p2_files": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(root.rglob("*.py"))
        },
    })


def protocol_id(config):
    semantic = deepcopy(config["p2"])
    for key in ("output_root", "p1_output_root", "progress_interval_seconds", "heartbeat_seconds"):
        semantic.pop(key)
    legacy = {key: value for key, value in legacy_config(config).items() if key not in ("execution", "assets")}
    return digest({"p2": semantic, "legacy": legacy, "code": code_id()})


def best_snapshots():
    return tuple(Snapshot(c.experts, c.k, s, "best") for c in cells() for s in SEEDS)


def cell_fields(experts, k):
    return {"E": experts, "k": k, "k_over_E": k / experts,
            "budget_tier": ("low", "mid", "high")[BUDGETS[experts].index(k)]}


def matrix():
    return {"cell_count": 9, "best_snapshots": 27, "new_training_runs": 0,
            "dense_traces": 1, "natural_r2_traces": 9, "natural_r4d_traces": 27,
            "new_full_evaluations": 81, "reused_p1_endpoints": 54,
            "local_conditions_per_snapshot": {"N2-a": 2, "N1": 10, "N3": 8}}
