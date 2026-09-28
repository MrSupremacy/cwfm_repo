from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import yaml


TASKS = ("sst2", "mnli", "qnli", "qqp")
ARMS = ("R2", "R4o", "R4d", "G1", "G2-0.001", "G4")
TRAINABLE_ARMS = frozenset(("R4o", "R4d", "G1", "G2-0.001", "G4"))
R4_FAMILY = frozenset(("R4o", "R4d"))
EXPERIMENT_BUDGETS = {
    64: (6, 10, 13),
    128: (13, 19, 26),
    256: (26, 38, 51),
}
FORMAL_EXPERTS = (128, 256)
CHECKPOINT_STEPS = tuple(range(0, 2641, 264))


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
        raise ValueError(f"extends must be a list of paths: {path}")
    result = {}
    for include in includes:
        result = merge(result, read_tree(path.parent / include, (*parents, path)))
    return merge(result, own)


def repository_root():
    return Path(__file__).resolve().parents[3]


def _path(config, key, fallback):
    value = Path(config["execution"].get(key, fallback)).expanduser()
    return (value if value.is_absolute() else repository_root() / value).resolve()


def root_for(config):
    return _path(config, "output_root", ".")


def input_root(config):
    return _path(config, "input_root", "inputs")


def tmp_root(config):
    return _path(config, "tmp_root", "tmp")


def dense_root(config):
    return root_for(config) / "dense_mt"


def routed_root(config):
    return root_for(config) / "routed_f0"


def budgets_for(config, experts):
    raw = config["experiment"]["budgets"]
    mapping = {int(key): tuple(int(k) for k in values) for key, values in raw.items()}
    try:
        return mapping[int(experts)]
    except KeyError as exc:
        raise ValueError(f"Unsupported expert count: {experts}") from exc


def suite_experts(config):
    if config["suite"]["name"] == "dense_mt":
        return ()
    return tuple(int(value) for value in config["suite"]["experts"])


def resolved_split(config, experts):
    experts = int(experts)
    d_ff = int(config["model"]["d_ff"])
    if experts not in EXPERIMENT_BUDGETS or d_ff % experts:
        raise ValueError(f"Unsupported equal-capacity split E={experts} for d_ff={d_ff}")
    size = d_ff // experts
    return {
        **config["split_generation"],
        "n_clusters": experts,
        "size_min": size,
        "size_max": size,
    }


def _validate_training_block(block, lr):
    expected = {
        "total_optimizer_steps": 2640,
        "checkpoint_every_steps": 264,
        "batch_size": 256,
        "domain_batch_size": 64,
        "accumulation_steps": 1,
        "warmup_steps": 132,
    }
    for key, value in expected.items():
        if int(block[key]) != value:
            raise ValueError(f"Formal {key} must be {value}")
    if block["optimizer"] != "Adam" or float(block["lr"]) != lr:
        raise ValueError("Optimizer/LR differs from the Phase D protocol")
    if block["amp"] or block["gradient_checkpointing"]:
        raise ValueError("AMP and gradient checkpointing are disabled")
    if float(block["max_grad_norm"]) != 1.0 or block["scheduler"] != "linear_to_zero":
        raise ValueError("Scheduler/clip differs from protocol")


def validate_config(config):
    suite = config["suite"]["name"]
    experiment = config["experiment"]
    budget_mapping = {
        int(key): tuple(int(k) for k in values)
        for key, values in experiment["budgets"].items()
    }
    if (
        experiment.get("phase") != "D"
        or experiment.get("finetune_mode") != "frozen"
        or experiment.get("split") != "balanced_kmeans"
        or tuple(experiment.get("experts", ())) != (64, 128, 256)
        or budget_mapping != EXPERIMENT_BUDGETS
    ):
        raise ValueError("Phase D E64/E128/E256 low-budget protocol is immutable")
    if tuple(config["tasks"]) != TASKS:
        raise ValueError("Task order must be sst2,mnli,qnli,qqp")
    if config["model"]["d_ff"] != 2048 or config["model"]["feed_forward_proj"] != "relu":
        raise ValueError("Only ReLU T5-small d_ff=2048 is supported")
    if config["model"]["precision"] != "float32" or config["model"]["tf32"]:
        raise ValueError("Formal execution is FP32 with TF32 disabled")
    if suite not in ("dense_mt", "phase_d_f0", "smoke"):
        raise ValueError("Supported suites: dense_mt, phase_d_f0, smoke")
    if suite != "smoke":
        if "dense" in config:
            _validate_training_block(config["dense"], 1.0e-5)
            if int(config["dense"]["seed"]) != 0:
                raise ValueError("Dense-MT seed is fixed to 0")
        if "training" in config:
            _validate_training_block(config["training"], 3.0e-4)
        if suite == "phase_d_f0":
            if suite_experts(config) != FORMAL_EXPERTS or config["suite"]["seeds"] != [0, 1, 2]:
                raise ValueError("Formal supplement is E128/E256 with routed seeds 0/1/2")
            if "budgets" in config["suite"]:
                raise ValueError("Formal supplement expands all three registered budgets per E")
    if suite in ("phase_d_f0", "smoke"):
        experts_values = suite_experts(config)
        if len(experts_values) != len(set(experts_values)) or any(value not in FORMAL_EXPERTS for value in experts_values):
            raise ValueError("New Phase D runs may use E128/E256 only; E64 is imported")
        variants = config["variants"]
        if [item["arm"] for item in variants] != list(ARMS):
            raise ValueError("Exactly six ordered Phase D arms are required")
        for item in variants:
            if bool(item["trainable"]) != (item["arm"] in TRAINABLE_ARMS):
                raise ValueError("Arm trainability mismatch")
        split = config["split_generation"]
        required = {
            "implementation": "kmeans_constrained", "k_means_constrained_version": "0.9.1",
            "scikit_learn_version": "1.9.0", "numpy_version": "2.4.3",
            "scipy_version": "1.18.0", "ortools_version": "9.15.6755",
            "joblib_version": "1.5.3", "normalize": "l2_rows", "random_state": 1,
            "init": "k-means++", "n_init": 10, "max_iter": 300,
            "tol": 1.0e-4, "copy_x": True, "n_jobs": 1,
        }
        for key, value in required.items():
            if split[key] != value:
                raise ValueError(f"Split parameter {key} differs from protocol")
        for experts, selected in config["suite"].get("budgets", {}).items():
            experts = int(experts)
            selected = tuple(int(k) for k in selected)
            if experts not in experts_values or not selected or len(selected) != len(set(selected)):
                raise ValueError("Invalid suite budget subset")
            if any(k not in budgets_for(config, experts) for k in selected):
                raise ValueError("Suite budget subset is outside protocol")
        for experts in experts_values:
            resolved = resolved_split(config, experts)
            if resolved["n_clusters"] != experts or resolved["size_min"] != 2048 // experts:
                raise ValueError("Split dimensions do not match the expert count")
    metrics = config["metrics"]
    if metrics["load_balance"] != ["cv"] or metrics["performance_comparison_role"] != "best":
        raise ValueError("CV-only and best-only comparisons are mandatory")
    forbidden = {"gini", "maximum_share", "coactivation", "coactivation_consistency"}
    tokens = json.dumps(config, sort_keys=True).lower()
    if any(name in tokens for name in forbidden):
        raise ValueError("Out-of-scope metric present")


def load_config(suite=None, local=None):
    root = repository_root()
    config = read_tree(suite or root / "configs/suites/phase_d_f0.yaml")
    if local:
        config = merge(config, read_tree(local))
    validate_config(config)
    return config


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=2)
def implementation_id(model_only=False):
    root = repository_root() / "src/task6_phased"
    excluded = {"metrics", "aggregation", "visualization"} if model_only else set()
    return digest({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if not excluded or path.relative_to(root).parts[0] not in excluded
    })


def protocol_id(config):
    relevant = {k: v for k, v in config.items() if k not in ("execution", "assets", "metrics")}
    return digest({"config": relevant, "implementation": implementation_id(True)})


def analysis_id(config):
    return digest({"metrics": config["metrics"], "implementation": implementation_id(False)})


@dataclass(frozen=True)
class Condition:
    experts: int
    arm: str
    k: int
    seed: int | None
    variant: str = "default"

    @property
    def trainable(self):
        return self.arm in TRAINABLE_ARMS

    @property
    def path(self):
        seed = self.seed if self.seed is not None else "deterministic"
        return Path(f"E_{self.experts}") / self.arm / self.variant / f"k_{self.k}" / f"seed_{seed}"

    def to_dict(self):
        return asdict(self)


def conditions(config):
    if config["suite"]["name"] == "dense_mt":
        return []
    seeds = config["suite"]["seeds"]
    result = []
    budget_subsets = config["suite"].get("budgets", {})
    for experts in suite_experts(config):
        budgets = budget_subsets.get(experts, budget_subsets.get(str(experts), budgets_for(config, experts)))
        for variant in config["variants"]:
            for k in budgets:
                for seed in seeds if variant["trainable"] else [None]:
                    result.append(Condition(experts, variant["arm"], int(k), seed, variant["name"]))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate conditions")
    return result


def variant_config(config, condition):
    return next(v for v in config["variants"] if (v["arm"], v["name"]) == (condition.arm, condition.variant))


def validate_run_id(run_id):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not run_id or any(c not in allowed for c in run_id):
        raise ValueError("run-id may contain only letters, digits, '_' and '-'")
    return run_id


def run_path(config, category, condition, run_id):
    validate_run_id(run_id)
    return routed_root(config) / "runs" / category / condition.path / run_id


def dense_run_path(config, category, run_id):
    validate_run_id(run_id)
    return dense_root(config) / "runs" / category / run_id


def expected_matrix_counts(config):
    if config["suite"]["name"] == "dense_mt":
        return {"dense_runs": 1, "dense_checkpoints": 11}
    items = conditions(config)
    trained = sum(c.trainable for c in items)
    static = len(items) - trained
    points = int(config["training"]["total_optimizer_steps"] // config["training"]["checkpoint_every_steps"]) + 1
    return {
        "conditions": len(items), "training_runs": trained, "static_states": static,
        "router_checkpoints": trained * points, "routed_states": trained * points + static,
    }


def recorded_protocol(config, condition, run_id):
    del condition, run_id
    return protocol_id(config)
