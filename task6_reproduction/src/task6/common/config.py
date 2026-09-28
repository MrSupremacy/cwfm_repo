from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import yaml


ARMS = ("R2", "R2-soft", "R4o", "R4d", "R4o-hard", "G1", "G2", "G4")
DENSE_INIT_ARM = "dense"
DENSE_FT_ARM = "dense-ft"
TRAINED_ARMS = frozenset((*ARMS, DENSE_FT_ARM))
ROUTER_PARAMETER_ARMS = frozenset(("R4o", "R4d", "R4o-hard", "G1", "G2", "G4"))
R4_FAMILY = frozenset(("R4o", "R4d", "R4o-hard"))
METRICS = ("performance", "load_balance", "churn", "oracle_overlap", "activation_coverage")


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader with duplicate and non-string key rejection."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError(f"Configuration keys must be strings: {key_node.start_mark}")
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
    config = read_tree(suite or root / "configs/suites/phase_a_f1.yaml")
    if local:
        config = merge(config, read_tree(local))
    validate_config(config)
    return config


def _variant_id(variant):
    return variant["arm"], variant["name"]


def validate_config(config):
    model, training, routing = config["model"], config["training"], config["routing"]
    suite, capture, metrics = config["suite"], config["capture"], config["metrics"]
    if (model["num_experts"], model["expert_size"], model["d_ff"]) != (64, 32, 2048):
        raise ValueError("Phase A requires 64 balanced experts of width 32")
    if tuple(suite["tasks"]) != ("sst2", "mnli") or tuple(suite["top_k"]) not in (
            (6, 13, 19, 26), (13,)):
        raise ValueError("Only the formal Phase A matrix or its k=13 smoke subset is supported")
    if suite.get("finetune_mode") != "F1" or training.get("mode") != "full":
        raise ValueError("This repository implements Phase A F1 full finetuning only")
    if training["world_size"] != 1 or training["accumulation_steps"] < 1:
        raise ValueError("Each run is single-GPU; accumulation must be positive")
    if training["amp"] or training["gradient_checkpointing"] or model["tf32"]:
        raise ValueError("Formal protocol uses FP32 with AMP, TF32 and gradient checkpointing disabled")
    if model["precision"] != "float32" or not config["execution"]["deterministic"]:
        raise ValueError("Deterministic FP32 execution is required")
    if training["optimizer"] != "Adam" or training["epochs"] != 10:
        raise ValueError("Expected Adam and ten complete epochs")
    if training["backbone_lr"] != 1.0e-5 or training["router_lr"] != 3.0e-4:
        raise ValueError("Phase A F1 uses backbone LR 1e-5 and router LR 3e-4")
    if training["checkpoint_payload"] != "full_training_state" or not (
            training["save_step_zero"] and training["retain_all_scheduled"]):
        raise ValueError("Every scheduled state must retain a complete training checkpoint")
    if routing != {
            "l2_epsilon": 1.0e-12, "rms_epsilon": 1.0e-6, "temperature": 1.0,
            "orthogonal_gain": 1.0, "bias_update_rate": 0.001,
            "tie_break": "score_desc_expert_id_asc", "centroid_refresh": "never",
            "hard_gradient_estimator": "coefficient_st_soft_backward"}:
        raise ValueError("Routing constants differ from the frozen Phase A protocol")
    expected_variants = {
        ("R2", "default"), ("R2-soft", "default"), ("R4o", "default"),
        ("R4d", "default"), ("R4o-hard", "default"), ("G1", "default"),
        ("G2", "aux_0.001"), ("G4", "default")}
    variants = [_variant_id(v) for v in config["variants"]]
    if len(variants) != len(set(variants)) or set(variants) != expected_variants:
        raise ValueError("Phase A F1 requires exactly the eight confirmed variants")
    g2 = next(v for v in config["variants"] if v["arm"] == "G2")
    if set(g2) != {"arm", "name", "aux_weight"} or g2["aux_weight"] != 0.001:
        raise ValueError("G2 must be the single alpha=0.001 variant")
    if tuple(suite["seeds"]) not in ((0, 1, 2), (0,)) or len(set(suite["seeds"])) != len(suite["seeds"]):
        raise ValueError("Formal seeds are 0/1/2; smoke may use seed 0")
    if suite.get("include_dense_reference") is not True:
        raise ValueError("Each task requires its fixed original-dense performance reference")
    if suite.get("include_dense_finetuned_baseline") is not True:
        raise ValueError("Each task requires the three-seed Dense-fullFT baseline")
    if suite["name"] == "phase_a_f1" and any(key in suite for key in ("train_limit", "validation_limit")):
        raise ValueError("Formal suite may not truncate data")
    if set(capture["parts"]) != {"A", "B", "C", "D"}:
        raise ValueError("Task 6 captures exactly A/B/C/D")
    forbidden_capture = {"coactivation_batch_size", "coactivation_chunk"} & set(capture)
    if forbidden_capture:
        raise ValueError(f"Co-activation capture is out of scope: {sorted(forbidden_capture)}")
    if capture["compression"] != "zstd" or capture["compression_level"] != 3:
        raise ValueError("Capture storage requires Parquet/ZSTD level 3")
    if capture["do_sample"] or capture["num_beams"] != 1:
        raise ValueError("Only greedy label generation is in scope")
    for key in ("generation_batch_size", "teacher_batch_size", "probe_batch_size", "shard_rows"):
        if capture[key] <= 0:
            raise ValueError(f"Invalid capture value: {key}")
    if tuple(metrics["enabled"]) != METRICS or metrics["load_balance"] != ["cv"]:
        raise ValueError("Metric scope is performance/CV/churn/overlap/coverage only")
    forbidden = {"random_seed", "random_repeats", "random_ratio_epsilon", "best_patched_gate"} & set(metrics)
    if forbidden:
        raise ValueError(f"Legacy metric settings are forbidden: {sorted(forbidden)}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=2)
def implementation_id(model_only=False):
    root = repository_root() / "src/task6"

    def include(path):
        relative = path.relative_to(root)
        return not model_only or relative.parts[0] not in ("metrics", "aggregation", "comparison", "visualization")

    return digest({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(root.rglob("*.py")) if include(path)
    })


def protocol_id(config):
    relevant = {key: value for key, value in config.items() if key not in ("execution", "assets", "metrics", "comparison")}
    return digest({"config": relevant, "implementation": implementation_id(model_only=True)})


def analysis_id(config):
    return digest({"metrics": config["metrics"], "implementation": implementation_id()})


@dataclass(frozen=True)
class Condition:
    task: str
    arm: str
    variant: str = "default"
    k: int = 0
    seed: int | None = None

    @property
    def trainable(self):
        return self.arm in TRAINED_ARMS

    @property
    def has_router_parameters(self):
        return self.arm in ROUTER_PARAMETER_ARMS

    @property
    def is_routed(self):
        return self.arm in ARMS

    @property
    def display_name(self):
        if self.arm == DENSE_INIT_ARM:
            return "Dense-init"
        if self.arm == DENSE_FT_ARM:
            return "Dense-fullFT"
        return "G2-0.001" if self.arm == "G2" else self.arm

    @property
    def path(self):
        seed = self.seed if self.seed is not None else "fixed"
        return Path(self.task) / self.arm / self.variant / f"k_{self.k}" / f"seed_{seed}"

    def to_dict(self):
        return asdict(self)


def conditions(config):
    suite = config["suite"]
    out = []
    if suite["include_dense_reference"]:
        out.extend(Condition(task, DENSE_INIT_ARM) for task in suite["tasks"])
    if suite["include_dense_finetuned_baseline"]:
        out.extend(Condition(task, DENSE_FT_ARM, seed=seed)
                   for task in suite["tasks"] for seed in suite["seeds"])
    for task in suite["tasks"]:
        for variant in config["variants"]:
            for k in suite["top_k"]:
                for seed in suite["seeds"]:
                    out.append(Condition(task, variant["arm"], variant["name"], k, seed))
    if len(out) != len(set(out)):
        raise ValueError("Duplicate experiment conditions")
    expected_train = len(suite["tasks"]) * len(suite["seeds"]) * (
        len(config["variants"]) * len(suite["top_k"]) + 1)
    if sum(c.trainable for c in out) != expected_train:
        raise ValueError("Condition expansion lost a fullFT training run")
    return out


def variant_config(config, condition):
    if not condition.is_routed:
        return {}
    return next(v for v in config["variants"] if _variant_id(v) == (condition.arm, condition.variant))


def root_for(config):
    return (repository_root() / config["execution"]["output_root"]).resolve()


def validate_run_id(run_id):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_id or any(ch not in allowed for ch in run_id):
        raise ValueError("run-id must contain only letters, digits, '_' or '-'")
    return run_id


def run_path(config, category, condition, run_id):
    validate_run_id(run_id)
    return root_for(config) / "runs" / category / condition.path / run_id
