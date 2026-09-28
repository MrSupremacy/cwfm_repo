from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import yaml


ARMS = ("R2", "R4o", "R4d", "G1", "G2-0.001", "G4")
TRAINABLE_ARMS = frozenset(("R4o", "R4d", "G1", "G2-0.001", "G4"))
R4_FAMILY = frozenset(("R4o", "R4d"))
EXPERIMENT_BUDGETS = {
    64: (6, 13, 19, 26),
    128: (13, 26, 38, 51),
    256: (26, 51, 77, 102),
}


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int)):
                raise ValueError(f"Configuration keys must be strings or integers: {key_node.start_mark}")
            if key in seen:
                raise ValueError(f"Duplicate configuration key {key!r}: {key_node.start_mark}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def merge(left, right):
    out = deepcopy(left)
    for key, value in right.items():
        out[key] = merge(out.get(key, {}), value) if isinstance(value, dict) else deepcopy(value)
    return out


def read_tree(path, parents=()):
    path = Path(path).resolve()
    if path in parents:
        raise ValueError(f"Cyclic configuration include: {path}")
    with path.open(encoding="utf-8") as stream:
        own = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(own, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    includes = own.pop("extends", [])
    if not isinstance(includes, list) or any(not isinstance(item, str) for item in includes):
        raise ValueError(f"extends must be a list of YAML paths: {path}")
    result = {}
    for include in includes:
        result = merge(result, read_tree(path.parent / include, (*parents, path)))
    return merge(result, own)


def repository_root():
    return Path(__file__).resolve().parents[3]


def load_config(suite=None, local=None):
    root = repository_root()
    config = read_tree(suite or root / "configs/suites/phase_b_f0.yaml")
    if local:
        config = merge(config, read_tree(local))
    validate_config(config)
    return config


def _budget_mapping(config):
    raw = config["experiment"]["budgets"]
    result = {int(experts): tuple(int(k) for k in values) for experts, values in raw.items()}
    return result


def budgets_for(config, experts):
    mapping = _budget_mapping(config)
    try:
        return mapping[int(experts)]
    except KeyError as exc:
        raise ValueError(f"Unsupported expert count: {experts}") from exc


def validate_config(config):
    experiment = config["experiment"]
    training = config["training"]
    routing = config["routing"]
    suite = config["suite"]
    metrics = config["metrics"]

    if experiment["phase"] != "B" or experiment["finetune_mode"] != "frozen":
        raise ValueError("This repository implements Task6 Phase B/F0 only")
    if experiment["split"] != "balanced_kmeans":
        raise ValueError("Phase B uses balanced parameter K-Means splits only")
    if tuple(experiment["all_experts"]) != (64, 128, 256) or _budget_mapping(config) != EXPERIMENT_BUDGETS:
        raise ValueError("The Phase B E/k matrix is fixed by the analysis protocol")
    if training["world_size"] != 1 or training["accumulation_steps"] != 1:
        raise ValueError("Each condition is an independent single-GPU run with accumulation=1")
    if training["amp"] or training["gradient_checkpointing"] or config["model"]["tf32"]:
        raise ValueError("AMP, gradient checkpointing and TF32 are disabled")
    if training["optimizer"] != "Adam" or not training["save_step_zero"]:
        raise ValueError("Expected Adam and a step-zero checkpoint")
    if suite["name"] == "phase_b_f0":
        if suite["tasks"] != ["sst2", "mnli"] or suite["experts"] != [128, 256] or suite["seeds"] != [0, 1, 2]:
            raise ValueError("Formal Phase B matrix must be 2 tasks x E128/E256 x 4 budgets x 3 seeds")
        if "budgets" in suite:
            raise ValueError("Formal Phase B always expands all four protocol budgets")
        if training["epochs"] != 10 or training["batch_size"] != 256 or "train_limit" in suite:
            raise ValueError("Formal training is 10 epochs, microbatch 256, without sample limits")
    elif suite["name"] != "smoke":
        raise ValueError("Only phase_b_f0 and smoke suites are supported")
    if len(suite["experts"]) != len(set(suite["experts"])) or any(e not in (128, 256) for e in suite["experts"]):
        raise ValueError("New runs may use E128/E256 only; E64 is imported")
    if len(suite["seeds"]) != len(set(suite["seeds"])):
        raise ValueError("Duplicate seeds")
    for experts, selected in suite.get("budgets", {}).items():
        experts = int(experts)
        selected = tuple(int(k) for k in selected)
        if experts not in suite["experts"] or not selected or len(selected) != len(set(selected)):
            raise ValueError("Invalid suite budget subset")
        if any(k not in budgets_for(config, experts) for k in selected):
            raise ValueError("Suite budget subset is outside the Phase B protocol")
    if config["model"]["d_ff"] != 2048 or any(config["model"]["d_ff"] % e for e in suite["experts"]):
        raise ValueError("Every expert count must exactly divide d_ff=2048")

    if routing != {
        "l2_epsilon": 1.0e-12,
        "rms_epsilon": 1.0e-6,
        "temperature": 1.0,
        "orthogonal_gain": 1.0,
        "bias_update_rate": 0.001,
        "tie_break": "score_desc_expert_id_asc",
    }:
        raise ValueError("Routing constants differ from the Phase B protocol")
    seen = set()
    for variant in config["variants"]:
        identity = (variant["arm"], variant["name"])
        if variant["arm"] not in ARMS or identity in seen:
            raise ValueError(f"Unknown or duplicate routing variant: {identity}")
        seen.add(identity)
        if variant["trainable"] != (variant["arm"] in TRAINABLE_ARMS):
            raise ValueError("Variant trainable flag conflicts with the arm")
        if variant["arm"] == "G2-0.001" and float(variant.get("aux_weight", -1)) != 0.001:
            raise ValueError("G2-0.001 must use auxiliary weight 0.001")
    if [variant["arm"] for variant in config["variants"]] != list(ARMS):
        raise ValueError("The suite must contain exactly the six ordered Phase B arms")

    expected_metrics = {
        "load_balance": ["cv"],
        "selection_quality": ["oracle_overlap", "adjusted_overlap", "activation_coverage"],
        "oracle_resolution": ["boundary_gap", "normalized_boundary_gap", "boundary_tie_rate", "all_zero_rate"],
        "stability": ["churn", "exact_set_change"],
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise ValueError(f"Metric scope mismatch for {key}")
    forbidden = {"gini", "maximum_share", "coactivation", "coactivation_consistency"}

    def all_strings(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from all_strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from all_strings(child)
        elif isinstance(value, str):
            yield value.lower()

    found = {token for token in all_strings(config) if token in forbidden}
    if found:
        raise ValueError(f"Out-of-scope metrics/configuration detected: {sorted(found)}")
    capture = config["capture"]
    if capture["compression"] != "zstd" or capture["compression_level"] != 3:
        raise ValueError("Capture storage is Parquet/ZSTD level 3")
    if capture["do_sample"] or capture["num_beams"] != 1:
        raise ValueError("Only greedy generation is allowed")
    for key in ("generation_batch_size", "teacher_batch_size", "probe_batch_size", "shard_rows"):
        if capture[key] <= 0:
            raise ValueError(f"Invalid capture parameter: {key}")
    if not config["execution"]["deterministic"] or config["model"]["precision"] != "float32":
        raise ValueError("Deterministic FP32 execution is required")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=2)
def implementation_id(model_only=False):
    root = repository_root() / "src/task6_phaseb"

    def include(path):
        relative = path.relative_to(root)
        return not model_only or relative.parts[0] not in ("metrics", "aggregation", "visualization", "imports")

    return digest({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(root.rglob("*.py")) if include(path)
    })


def protocol_id(config):
    relevant = {key: value for key, value in config.items() if key not in ("execution", "assets", "imports", "metrics")}
    return digest({"config": relevant, "implementation": implementation_id(model_only=True)})


def analysis_id(config):
    return digest({"metrics": config["metrics"], "implementation": implementation_id()})


@dataclass(frozen=True)
class Condition:
    task: str
    experts: int
    arm: str
    variant: str = "default"
    k: int = 0
    seed: int | None = None

    @property
    def trainable(self):
        return self.arm in TRAINABLE_ARMS

    @property
    def path(self):
        seed = self.seed if self.seed is not None else "fixed"
        return Path(self.task) / f"E_{self.experts}" / self.arm / self.variant / f"k_{self.k}" / f"seed_{seed}"

    def to_dict(self):
        return asdict(self)


def conditions(config):
    suite = config["suite"]
    result = []
    for task in suite["tasks"]:
        for experts in suite["experts"]:
            for variant in config["variants"]:
                seeds = suite["seeds"] if variant["trainable"] else [None]
                selected_budgets = config["suite"].get("budgets", {}).get(
                    experts, config["suite"].get("budgets", {}).get(str(experts), budgets_for(config, experts))
                )
                for k in selected_budgets:
                    for seed in seeds:
                        result.append(Condition(task, experts, variant["arm"], variant["name"], k, seed))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate experiment conditions")
    return result


def variant_config(config, condition):
    return next(
        variant for variant in config["variants"]
        if (variant["arm"], variant["name"]) == (condition.arm, condition.variant)
    )


def root_for(config):
    path = Path(config["execution"]["output_root"]).expanduser()
    if not path.is_absolute():
        path = repository_root() / path
    return path.resolve()


def validate_run_id(run_id):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_id or any(character not in allowed for character in run_id):
        raise ValueError("run-id must contain only letters, digits, '_' or '-'")
    return run_id


def run_path(config, category, condition, run_id):
    validate_run_id(run_id)
    return root_for(config) / "runs" / category / condition.path / run_id


def recorded_protocol(config, condition, run_id):
    del condition, run_id
    return protocol_id(config)


def expected_matrix_counts(config):
    items = conditions(config)
    trained = sum(condition.trainable for condition in items)
    static = len(items) - trained
    states = trained * (config["training"]["epochs"] + 1) + static
    return {"conditions": len(items), "training_runs": trained, "static_states": static, "routed_states": states}
