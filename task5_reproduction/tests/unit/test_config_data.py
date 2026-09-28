from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import subprocess
import sys
import json
import numpy as np

from task5.common.config import (Condition, analysis_id, conditions, digest, extension_spec, load_config,
                                 protocol_id, read_tree, recorded_protocol, repository_root,
                                 validate_config, validate_run_id)
from task5.data.datasets import cache_declaration, expected_probe_keys, probe_members, tokenize
from task5.training.schedule import scheduler_scale


class FakeTokenizer:
    padding_side = truncation_side = "right"
    special_tokens_map = {"eos_token": "</s>"}

    @property
    def backend_tokenizer(self):
        return self

    def to_str(self):
        return json.dumps({"model": {"vocab": {"hello": 0}}, "padding": None, "truncation": None})

    def __call__(self, text=None, text_target=None, **kwargs):
        values = text if text_target is None else text_target
        return {"input_ids": [[2, 1] for _ in values], "attention_mask": [[1, 1] for _ in values]}


class FakeDataset:
    _fingerprint = "fixture_raw_v1"
    column_names = ["sentence", "label"]

    def __len__(self):
        return 2

    def map(self, fn, **kwargs):
        self.options = kwargs
        return fn({"sentence": ["first", "second"], "label": [0, 1]}, [0, 1])


class ConfigDataTests(TestCase):
    def setUp(self):
        self.config = load_config()

    def test_main_matrix_counts(self):
        matrix = conditions(self.config)
        trained = [c for c in matrix if c.trainable]
        self.assertEqual(len(trained), 192)
        self.assertEqual(len(matrix), 242)
        self.assertEqual(len(trained) * 11 + len(matrix) - len(trained), 2162)
        self.assertEqual(len([c for c in matrix if c.arm == "G2"]), 72)
        r4_r2init = [c for c in matrix if c.arm == "R4-R2Init"]
        self.assertEqual(len(r4_r2init), 24)
        self.assertTrue(all(c.trainable and c.path.parts[1] == "R4-R2Init" for c in r4_r2init))
        self.assertTrue(all(c.seed is None for c in matrix if c.arm in ("dense", "R1", "R2", "R3")))

    def test_smoke_matrix(self):
        c = load_config(repository_root() / "configs/suites/smoke.yaml")
        self.assertEqual(sum(x.trainable for x in conditions(c)), 16)
        self.assertEqual(c["suite"]["train_limit"], 512)
        self.assertEqual(c["suite"]["validation_limit"], 32)

    def test_latest_r3_decision(self):
        self.assertEqual(self.config["routing"]["rms_epsilon"], 1e-6)
        self.assertFalse(self.config["routing"]["r2_r3_equal_sets_required"])

    def test_unsupported_protocol_changes_fail(self):
        cases = [("training", "amp", True), ("training", "world_size", 8), ("model", "tf32", True),
                 ("training", "accumulation_steps", 4), ("metrics", "best_patched_gate", "auto"),
                 ("routing", "r2_r3_equal_sets_required", True)]
        for section, key, value in cases:
            c = deepcopy(self.config)
            c[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(c)

    def test_run_ids_cannot_escape(self):
        for text in ("../elsewhere", "a/b", "", "..", "a\\b"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_run_id(text)
        self.assertEqual(validate_run_id("main_01-retry"), "main_01-retry")

    def test_machine_paths_do_not_change_protocol(self):
        c = deepcopy(self.config)
        c["execution"]["output_root"] = "/another/server"
        c["assets"] = {"sst2": {"dense": "/elsewhere"}}
        self.assertEqual(protocol_id(c), protocol_id(self.config))
        c["capture"]["probe_batch_size"] = 128
        self.assertNotEqual(protocol_id(c), protocol_id(self.config))

    def test_explicit_extension_keeps_base_and_new_protocols_distinct(self):
        c = deepcopy(self.config)
        base = "a" * 64
        c["extension"] = {"arm": "R4-R2Init", "base_run_id": "formal01", "base_protocol": base}
        validate_config(c)
        self.assertEqual(extension_spec(c, "formal01")["base_protocol"], base)
        self.assertEqual(recorded_protocol(c, Condition("sst2", "dense"), "formal01"), base)
        new = Condition("sst2", "R4-R2Init", "default", 6, 0)
        self.assertEqual(recorded_protocol(c, new, "formal01"), protocol_id(c))
        self.assertNotEqual(protocol_id(c), base)
        with self.assertRaises(ValueError):
            extension_spec(c, "another_run")
        c["extension"]["base_protocol"] = "not-a-digest"
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_config_include_cycle_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "cycle.yaml"
            path.write_text('extends: [cycle.yaml]\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_tree(path)

    def test_offline_config_does_not_invalidate_captures(self):
        c = deepcopy(self.config)
        c["metrics"]["chunk_rows"] = 4096
        self.assertEqual(protocol_id(c), protocol_id(self.config))
        self.assertNotEqual(analysis_id(c), analysis_id(self.config))

    def test_yaml_float_and_boolean_types(self):
        for section, names in (("routing", ("l2_epsilon", "rms_epsilon", "noise_epsilon")),
                               ("training", ("eps", "lr")), ("metrics", ("random_ratio_epsilon",))):
            for name in names:
                self.assertIs(type(self.config[section][name]), float)
        self.assertIs(type(self.config["training"]["amp"]), bool)
        self.assertIs(type(self.config["suite"]["seeds"][0]), int)

    def test_yaml_inheritance_and_local_override(self):
        with TemporaryDirectory() as temp:
            base = Path(temp) / "base.yaml"
            child = Path(temp) / "child.yaml"
            base.write_text('nested: {keep: 1, replace: 2}\nitems: [1, 2]\n', encoding="utf-8")
            child.write_text('extends: [base.yaml]\nnested: {replace: 3}\nitems: [4]\n', encoding="utf-8")
            self.assertEqual(read_tree(child), {"nested": {"keep": 1, "replace": 3}, "items": [4]})
        c = load_config(local=repository_root() / "configs/local/server.yaml")
        self.assertEqual(c["assets"]["sst2"]["dense"], "inputs/dense/sst2")

    def test_yaml_invalid_structure_and_duplicates_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            for content in ('', '[]', 'extends: base.yaml', 'extends: [3]',
                            'x: 1\nx: 2', 'x: {a: 1, a: 2}', '1: value'):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_tree(path)

    def test_yaml_unsafe_python_objects_rejected(self):
        import yaml
        with TemporaryDirectory() as temp:
            path = Path(temp) / "unsafe.yaml"
            path.write_text('value: !!python/tuple [1, 2]', encoding="utf-8")
            with self.assertRaises(yaml.constructor.ConstructorError):
                read_tree(path)

    def test_stratified_sampling_largest_remainder(self):
        labels = np.array([0] * 5 + [1] * 3 + [2] * 2)
        ids = probe_members(labels, 7)
        self.assertEqual(np.bincount(labels[ids]).tolist(), [4, 2, 1])
        self.assertEqual(len(set(ids)), 7)
        np.testing.assert_array_equal(ids, np.sort(ids))
        np.testing.assert_array_equal(ids, probe_members(labels, 7))
        np.testing.assert_array_equal(probe_members(labels, 10), np.arange(10))

    def test_sampling_tie_uses_label_id(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(np.bincount(labels[probe_members(labels, 4)]), [2, 1, 1])

    def test_cache_identity_changes_with_inputs_and_protocol(self):
        ds, tok = FakeDataset(), FakeTokenizer()
        first = cache_declaration(self.config, "sst2", ds, tok, {"dense": "A", "data": "D"}, "validation")
        second = cache_declaration(self.config, "sst2", ds, tok, {"dense": "B", "data": "D"}, "validation")
        self.assertNotEqual(digest(first), digest(second))
        c = deepcopy(self.config)
        c["data"]["max_source_length"] = 64
        self.assertNotEqual(digest(first), digest(cache_declaration(c, "sst2", ds, tok, {"dense": "A", "data": "D"}, "validation")))

    def test_cache_stays_inside_output_and_keeps_sample_ids(self):
        with TemporaryDirectory() as temp:
            c = deepcopy(self.config)
            c["execution"]["output_root"] = temp
            ds = FakeDataset()
            result = tokenize(c, "sst2", ds, FakeTokenizer(), {"data": "fixture"}, "validation")
            self.assertEqual(result["sample_id"], [0, 1])
            cache = Path(ds.options["cache_file_name"])
            self.assertTrue(cache.is_relative_to((Path(temp) / "tmp/preprocessing").resolve()))
            self.assertEqual(len(cache.parent.name), 64)
            self.assertEqual(ds.options["new_fingerprint"], cache.parent.name[:32])
            self.assertLessEqual(len(ds.options["new_fingerprint"] + "_00000_of_00004"), 64)
            self.assertTrue(ds.options["load_from_cache_file"])

    def test_expected_probe_keys_respect_padding(self):
        rows = [{"sample_id": 3, "input_ids": [2, 3], "labels": [7]},
                {"sample_id": 8, "input_ids": [4], "labels": [7, 1]}]
        a = expected_probe_keys(rows, 2, "right")
        b = expected_probe_keys(rows, 2, "left")
        self.assertEqual(a["encoder"]["count"], 3)
        self.assertEqual(a["decoder"]["count"], 3)
        self.assertNotEqual(a["encoder"]["sha256"], b["encoder"]["sha256"])

    def test_scheduler_bounds_and_documented_steps(self):
        import math
        for n, total, warmup in ((67349, 2640, 132), (392702, 15340, 767)):
            self.assertEqual(math.ceil(n / 256) * 10, total)
            self.assertEqual(math.ceil(total * .05), warmup)
            self.assertEqual(scheduler_scale(0, total, warmup), 0)
            self.assertEqual(scheduler_scale(warmup, total, warmup), 1)
            self.assertEqual(scheduler_scale(total, total, warmup), 0)

    def test_offline_imports_do_not_load_model_libraries(self):
        code = ("import sys; sys.path.insert(0, " + repr(str(repository_root() / "src")) + "); "
                "import task5.metrics.pipeline, task5.aggregation.pipeline, task5.visualization.render; "
                "assert not ({'torch','transformers','datasets'} & set(sys.modules)); print('offline-imports-ok')")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertIn("offline-imports-ok", result.stdout)
