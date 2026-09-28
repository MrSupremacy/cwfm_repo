from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import yaml

ARMS = ("R1", "R2", "R2-soft", "R3", "R4", "R4-R2Init", "R4-hard", "G0", "G1", "G2", "G3", "G4")
TRAINABLE_ARMS = frozenset(("R4", "R4-R2Init", "R4-hard", "G1", "G2", "G3", "G4"))
R4_FAMILY = frozenset(("R4", "R4-R2Init", "R4-hard"))


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML with explicit duplicate-key rejection (no silent overrides)."""

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
    if not isinstance(includes, list) or any(not isinstance(p, str) for p in includes):
        raise ValueError(f"extends must be a list of YAML paths: {path}")
    result = {}
    for include in includes:
        result = merge(result, read_tree(path.parent / include, (*parents, path)))
    return merge(result, own)


def repository_root():
    return Path(__file__).resolve().parents[3]


def load_config(suite=None, local=None):
    root = repository_root()
    config = read_tree(suite or root / "configs/suites/main.yaml")
    if local:
        config = merge(config, read_tree(local))
    validate_config(config)
    return config


def validate_config(c):
    t, r = c["training"], c["routing"]
    if t["world_size"] != 1 or t["accumulation_steps"] != 1:
        raise ValueError("This protocol supports one GPU/run and accumulation=1 only.")
    if t["amp"] or t["gradient_checkpointing"] or c["model"]["tf32"]:
        raise ValueError("AMP, activation checkpointing and TF32 are disabled by protocol.")
    if t["optimizer"] != "Adam" or t["epochs"] != 10 or not t["save_step_zero"]:
        raise ValueError("Expected Adam, 10 epochs, and initialization checkpoint.")
    if r["r3"] != "frozen_raw_centroid_rms_hard" or r["r2_r3_equal_sets_required"]:
        raise ValueError("Latest user decision retains RMS R3 without exact R2 agreement gate.")
    if c["metrics"]["best_patched_gate"] != "disabled_pending_M19":
        raise ValueError("M19 has not been authorized; report named candidates separately.")
    for key in ("batch_size",):
        if t[key] <= 0:
            raise ValueError(f"Invalid training {key}")
    if c["suite"]["name"] == "main" and any(k in c["suite"] for k in ("train_limit", "validation_limit")):
        raise ValueError("Sample limits are forbidden in the main suite.")
    if len(c["suite"]["seeds"]) != len(set(c["suite"]["seeds"])):
        raise ValueError("Duplicate seeds")
    if not c["suite"]["include_dense"]:
        raise ValueError("Dense A is required as the relative-performance denominator")
    if any(k not in (6, 13, 19, 26) for k in c["suite"]["top_k"]):
        raise ValueError("Only the four confirmed budgets belong to the experiment suites")
    if len(c["suite"]["top_k"]) != len(set(c["suite"]["top_k"])):
        raise ValueError("Duplicate budgets")
    if c["capture"]["compression"] != "zstd" or c["capture"]["compression_level"] != 3:
        raise ValueError("Capture storage is Parquet/ZSTD level 3")
    if c["capture"]["do_sample"] or c["capture"]["num_beams"] != 1:
        raise ValueError("Only greedy label generation is in scope")
    for key in ("generation_batch_size", "teacher_batch_size", "probe_batch_size", "coactivation_batch_size", "coactivation_chunk", "shard_rows"):
        if c["capture"][key] <= 0:
            raise ValueError(f"Invalid capture {key}")
    if not c["execution"]["deterministic"] or c["model"]["precision"] != "float32":
        raise ValueError("This implementation requires deterministic FP32 execution")
    seen = set()
    for v in c["variants"]:
        identity = (v["arm"], v["name"])
        if v["arm"] not in ARMS:
            raise ValueError(f"Unknown routing arm: {v['arm']}")
        if identity in seen:
            raise ValueError(f"Duplicate routing variant: {identity}")
        seen.add(identity)
        if v["trainable"] != (v["arm"] in TRAINABLE_ARMS):
            raise ValueError("Variant trainable flag conflicts with its arm")
    if ("R4-R2Init", "default") not in seen:
        raise ValueError("The confirmed R4-R2Init extension is required")
    extension = c.get("extension")
    if extension is not None:
        legacy = set(extension) == {"arm", "base_run_id", "base_protocol"}
        phase_a = set(extension) == {"arms", "base_run_id", "base_protocol", "source_report"}
        if legacy:
            if extension["arm"] != "R4-R2Init":
                raise ValueError("Legacy extension supports only R4-R2Init")
        elif phase_a:
            if extension["arms"] != ["R2-soft", "R4-hard"]:
                raise ValueError("Phase A F0 extends exactly R2-soft and R4-hard")
            if any((arm, "default") not in seen for arm in extension["arms"]):
                raise ValueError("New arms must be present in the Phase A F0 suite")
            source = extension["source_report"]
            if set(source) != {"protocol", "analysis", "sha256"}:
                raise ValueError("source_report requires protocol/analysis/sha256")
            for value in source.values():
                if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                    raise ValueError("source_report identities must be SHA256 digests")
        else:
            raise ValueError("Unsupported extension declaration")
        validate_run_id(extension["base_run_id"])
        value = extension["base_protocol"]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("extension.base_protocol must be a lowercase SHA256 digest")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=2)
def implementation_id(model_only=False):
    root = repository_root() / "src/task5"
    def include(path):
        relative = path.relative_to(root)
        # Offline metric/plot changes must not invalidate expensive model captures.
        return not model_only or relative.parts[0] not in ("metrics", "aggregation", "visualization")
    # Git checkouts on Windows/Linux may differ only by line endings. New
    # protocols use canonical source bytes; historical protocols stay explicit.
    return digest({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                   for p in sorted(root.rglob("*.py")) if include(p)})


def protocol_id(config):
    # Machine path/device changes are recorded separately through input identities.
    relevant = {k: v for k, v in config.items() if k not in ("execution", "assets", "metrics")}
    return digest({"config": relevant, "implementation": implementation_id(model_only=True)})


def analysis_id(config):
    return digest({"metrics": config["metrics"], "implementation": implementation_id()})


def extension_spec(config, run_id):
    """Return an explicitly configured, run-bound extension declaration."""
    extension = config.get("extension")
    if extension is None:
        return None
    if run_id != extension["base_run_id"]:
        raise ValueError(
            f"Extension configuration is bound to run-id {extension['base_run_id']!r}, not {run_id!r}"
        )
    return extension


def recorded_protocol(config, condition, run_id):
    """Protocol expected for a record read during an extension run.

    The new arm writes under the current protocol. Existing arms are read-only
    dependencies and retain the verified base protocol recorded at creation.
    """
    extension = extension_spec(config, run_id)
    if extension is not None and condition.arm not in extension_arms(extension):
        return extension["base_protocol"]
    return protocol_id(config)


def extension_arms(extension):
    return extension.get("arms", [extension.get("arm")])


@dataclass(frozen=True)
class Condition:
    task: str
    arm: str
    variant: str = "default"
    k: int = 0
    seed: int | None = None

    @property
    def trainable(self):
        return self.arm in TRAINABLE_ARMS

    @property
    def path(self):
        return Path(self.task) / self.arm / self.variant / f"k_{self.k}" / f"seed_{self.seed if self.seed is not None else 'fixed'}"

    def to_dict(self):
        return asdict(self)


def conditions(config):
    s = config["suite"]
    out = []
    for task in s["tasks"]:
        if s["include_dense"]:
            out.append(Condition(task, "dense"))
        for variant in config["variants"]:
            seeds = s["seeds"] if variant["trainable"] or variant["arm"] == "G0" else [None]
            for k in s["top_k"]:
                for seed in seeds:
                    out.append(Condition(task, variant["arm"], variant["name"], k, seed))
    if len(set(out)) != len(out):
        raise ValueError("Duplicate experiment conditions")
    return out


def variant_config(config, condition):
    return next(v for v in config["variants"] if (v["arm"], v["name"]) == (condition.arm, condition.variant))


def root_for(config):
    return (repository_root() / config["execution"]["output_root"]).resolve()


def validate_run_id(run_id):
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
        raise ValueError("run-id must contain only letters, digits, '_' or '-'")
    return run_id


def run_path(config, category, condition, run_id):
    validate_run_id(run_id)
    return root_for(config) / "runs" / category / condition.path / run_id
