from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import yaml


TASKS = ("sst2", "mnli", "qnli", "qqp")
MODES = ("M00", "M01", "M10", "M11")
EXPERTS = (64, 128, 256)
BUDGETS = {64: (6, 10, 13), 128: (13, 19, 26), 256: (26, 38, 51)}
SEEDS = (0, 1, 2)


class ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int)) or key in seen:
                raise ValueError(f"Invalid or duplicate YAML key {key!r}: {key_node.start_mark}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def merge(left, right):
    result = deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else deepcopy(value)
    return result


def read_tree(path, parents=()):
    path = Path(path).resolve()
    if path in parents:
        raise ValueError(f"Cyclic configuration include: {path}")
    with path.open(encoding="utf-8") as stream:
        own = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(own, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    includes = own.pop("extends", [])
    if not isinstance(includes, list) or any(not isinstance(item, str) for item in includes):
        raise ValueError(f"extends must be a list of paths: {path}")
    result = {}
    for include in includes:
        result = merge(result, read_tree(path.parent / include, (*parents, path)))
    return merge(result, own)


def repository_root():
    return Path(__file__).resolve().parents[3]


def resolve_path(config, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository_root() / path).resolve()


def validate_config(config):
    protocol = config["protocol"]
    budgets = {int(e): tuple(int(k) for k in ks) for e, ks in protocol["budgets"].items()}
    expected = {
        "source_phase": "task6_phaseD_F0",
        "source_run_id": "routed02",
        "checkpoint_roles": ["init", "best"],
        "tasks": list(TASKS),
        "experts": list(EXPERTS),
        "seeds": list(SEEDS),
        "modes": list(MODES),
    }
    for key, wanted in expected.items():
        if protocol.get(key) != wanted:
            raise ValueError(f"Frozen P0/P1 protocol mismatch at protocol.{key}")
    if budgets != BUDGETS:
        raise ValueError("The D9 E/k matrix is immutable")
    routing = config["routing"]
    if (
        float(routing["l2_epsilon"]) != 1.0e-12
        or float(routing["rms_epsilon"]) != 1.0e-6
        or float(routing["temperature"]) != 1.0
        or routing["tie_break"] != "score_desc_expert_id_asc"
        or routing["weight_scale"] != "k"
    ):
        raise ValueError("Routing numerics differ from the Task 6 endpoints")
    diagnostic = config["diagnostic"]
    if (
        diagnostic["name"] != "diagnostic_128"
        or int(diagnostic["count_per_task"]) != 128
        or int(diagnostic["seed"]) != 0
    ):
        raise ValueError("diagnostic_128 definition is immutable")
    if config["suite"]["checkpoint_roles"] != ["init", "best"]:
        raise ValueError("This implementation manifest requires init and best")


def load_config(suite=None, local=None):
    root = repository_root()
    value = read_tree(suite or root / "configs/suites/p01_init_best.yaml")
    if local:
        value = merge(value, read_tree(local))
    validate_config(value)
    return value


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def implementation_id():
    root = repository_root() / "src/task8_p01"
    return digest({
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in sorted(root.rglob("*.py"))
    })


def protocol_id(config):
    stable = {key: value for key, value in config.items() if key not in ("execution", "assets")}
    return digest({"config": stable, "implementation": implementation_id()})


@dataclass(frozen=True)
class Cell:
    experts: int
    k: int

    def __post_init__(self):
        if self.experts not in BUDGETS or self.k not in BUDGETS[self.experts]:
            raise ValueError(f"Outside D9: E={self.experts}, k={self.k}")

    @property
    def path(self):
        return Path(f"E_{self.experts}") / f"k_{self.k}"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Snapshot:
    experts: int
    k: int
    seed: int | None
    role: str = "best"

    def __post_init__(self):
        Cell(self.experts, self.k)
        if self.role not in ("init", "best"):
            raise ValueError("P0/P1 snapshot role must be init or best")
        if self.role == "init" and self.seed is not None:
            raise ValueError("Init is deduplicated across seeds and must use seed=None")
        if self.role == "best" and self.seed not in SEEDS:
            raise ValueError("Best snapshots require seed 0/1/2")

    @property
    def path(self):
        if self.role == "init":
            return Cell(self.experts, self.k).path / "init"
        return Cell(self.experts, self.k).path / f"seed_{self.seed}" / "best"

    def to_dict(self):
        return asdict(self)


def cells():
    return tuple(Cell(e, k) for e in EXPERTS for k in BUDGETS[e])


def snapshots():
    return tuple(
        [Snapshot(cell.experts, cell.k, None, "init") for cell in cells()]
        + [Snapshot(cell.experts, cell.k, seed, "best") for cell in cells() for seed in SEEDS]
    )


def matrix_counts():
    return {
        "training_runs": 0,
        "cells": len(cells()),
        "r4d_init_snapshots": len(cells()),
        "r4d_best_snapshots": len(cells()) * len(SEEDS),
        "r4d_total_snapshots": len(snapshots()),
        "M00_unique_evaluations_per_population": len(cells()),
        "learned_mode_evaluations_per_population": len(snapshots()) * 3,
        "four_cell_rows_per_population": len(cells()) + len(snapshots()) * 3,
        "populations": 2,
    }
