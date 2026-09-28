from __future__ import annotations

from functools import cached_property

from task8_p01.assets.catalog import load_summaries
from task8_p01.common.config import TASKS, digest
from task8_p01.common.io import checked_complete, read_json
from task8_p01.compat.task6 import load_task6_config, task6_source_identity
from task8_p2.config import code_id, legacy_config, output_root, p1_root, protocol_id
from task8_p2.progress import heartbeat


class Context:
    def __init__(self, config, *, smoke=False):
        self.config, self.smoke = config, bool(smoke)
        self.legacy = legacy_config(config)
        # Read existing P1 assets/panel only. Never freeze a new panel in P1.
        self.legacy["execution"]["output_root"] = str(p1_root(config))
        self.task6 = load_task6_config(self.legacy)
        self.task6["execution"]["device"] = config["execution"]["device"]
        self.device = self.task6["execution"]["device"]
        self._assets, self._datasets = {}, {}
        self.verified = set()
        import torch
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    def assets(self, experts):
        if experts not in self._assets:
            from task6_phased.substrate.assets import inspect_assets
            with heartbeat(f"verify assets E={experts}", self.config["p2"]["heartbeat_seconds"]):
                self._assets[experts] = inspect_assets(self.task6, experts)
        return self._assets[experts]

    @cached_property
    def source_identity(self):
        return task6_source_identity(self.legacy)

    @cached_property
    def panel(self):
        root = p1_root(self.config) / "artifacts/diagnostic_128"
        header = checked_complete(root)
        value = read_json(root / "manifest.json")
        if header["protocol"] != self.config["p2"]["expected_p1_protocol"]:
            raise ValueError("The diagnostic panel is not from the registered P1 protocol")
        if value["count_per_task"] != 128 or value["seed"] != 0 or set(value["tasks"]) != set(TASKS):
            raise ValueError("Invalid diagnostic_128 panel")
        expected = digest({t: value["tasks"][t]["hash"] for t in TASKS})
        if value["panel_hash"] != expected:
            raise ValueError("Panel task hashes do not match panel_hash")
        return value

    def load_model(self, experts=64):
        from task6_phased.dense.model import load_t5
        paths = self.assets(experts)[0]
        with heartbeat("load frozen Dense-MT model", self.config["p2"]["heartbeat_seconds"]):
            return load_t5(self.task6, paths["dense_best"], trainable=False)

    def datasets(self, tokenizer, population="diagnostic_128"):
        if population in self._datasets:
            return self._datasets[population]
        from task6_phased.data.datasets import load_raw, tokenize
        dense_hash = self.assets(64)[3]["dense"]["model_sha256"]
        result = {}
        with heartbeat(f"load/tokenize {population}", self.config["p2"]["heartbeat_seconds"]):
            for task in TASKS:
                _, raw = load_raw(self.task6, task)
                encoded = tokenize(self.task6, task, raw, tokenizer, "validation", dense_hash)
                if population == "diagnostic_128":
                    self.panel
                    value = read_json(p1_root(self.config) / f"artifacts/diagnostic_128/{task}.json")
                    members = value["sample_ids"]
                    label_rows = [{"task": task, "source_index": i, "label": int(raw[i]["label"])} for i in members]
                    if len(members) != 128 or digest(label_rows) != value["hash"]:
                        raise ValueError(f"Panel membership/data changed for {task}")
                    encoded = encoded.select(members)
                if self.smoke:
                    encoded = encoded.select(range(min(len(encoded), int(self.config["p2"]["smoke_samples_per_task"]))))
                result[task] = encoded
        self._datasets[population] = result
        return result

    def data_identity(self, population):
        value = {
            "parquet_sha256": {t: self.task6["data"]["parquet_sha256"][t]["validation"] for t in TASKS},
            "tasks": self.task6["tasks"],
            "source_length": self.task6["data"]["max_source_length"],
            "target_length": self.task6["data"]["max_target_length"],
            "dense_hash": self.assets(64)[3]["dense"]["model_sha256"],
            "population": population,
            "panel_hash": self.panel["panel_hash"] if population == "diagnostic_128" else None,
            "smoke": self.smoke,
            "smoke_limit": self.config["p2"]["smoke_samples_per_task"] if self.smoke else None,
        }
        return {"data_role": f"smoke_{population}" if self.smoke else population, "data_hash": digest(value), "data_definition": value}

    def identity(self, population="diagnostic_128"):
        return {"schema": 1, "phase": "P2-N", "protocol": protocol_id(self.config),
                "code_hash": code_id(), "task6_source_hash": self.source_identity["sha256"],
                **self.data_identity(population)}

    def snapshot(self, snapshot):
        with heartbeat(f"verify best E{snapshot.experts}/k{snapshot.k}/seed{snapshot.seed}", self.config["p2"]["heartbeat_seconds"]):
            learned, checkpoint = load_summaries(self.legacy, snapshot)
        return learned, checkpoint

    def namespace(self):
        population = self.data_identity("diagnostic_128")["data_role"]
        return output_root(self.config) / "caches" / protocol_id(self.config)[:16] / population
