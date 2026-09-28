from copy import deepcopy
from unittest import TestCase

import numpy as np

from task6_phaseb.aggregation.core import layer_summary, seed_summary
from task6_phaseb.aggregation.pipeline import gap_changes, paired_differences
from task6_phaseb.capture.storage import validate_selection
from task6_phaseb.common.config import ARMS, Condition, conditions, expected_matrix_counts, load_config, validate_config
from task6_phaseb.imports.phase_a_f0 import _add_adjusted_overlap, _convert_row
from task6_phaseb.metrics.load_balance.core import load_metrics
from task6_phaseb.metrics.performance.core import choose_best, performance
from task6_phaseb.metrics.selection_quality.core import selection_statistics
from task6_phaseb.metrics.stability.core import churn
from task6_phaseb.substrate.assets import validate_labels


class PhaseBProtocolTests(TestCase):
    def test_formal_matrix_is_exact(self):
        config = load_config()
        self.assertEqual(tuple(item["arm"] for item in config["variants"]), ARMS)
        self.assertEqual(expected_matrix_counts(config), {
            "conditions": 256, "training_runs": 240, "static_states": 16, "routed_states": 2656,
        })
        items = conditions(config)
        self.assertEqual({c.experts for c in items}, {128, 256})
        self.assertEqual({c.k for c in items if c.experts == 128}, {13, 26, 38, 51})
        self.assertEqual({c.k for c in items if c.experts == 256}, {26, 51, 77, 102})
        self.assertTrue(all(c.seed is None for c in items if c.arm == "R2"))
        self.assertTrue(all(c.seed in (0, 1, 2) for c in items if c.arm != "R2"))

    def test_condition_identity_contains_expert_count(self):
        value = Condition("mnli", 256, "G4", "default", 51, 2)
        self.assertEqual(
            value.path.as_posix(),
            "mnli/E_256/G4/default/k_51/seed_2",
        )

    def test_out_of_scope_metric_is_rejected(self):
        config = load_config()
        for forbidden in ("gini", "maximum_share", "coactivation"):
            changed = deepcopy(config)
            changed["metrics"]["extra"] = [forbidden]
            with self.assertRaisesRegex(ValueError, "Out-of-scope"):
                validate_config(changed)

    def test_performance_cv_and_seed_statistics(self):
        records = {
            "sample_id": [0, 1, 2, 3],
            "prediction_right": [True, False, True, False],
            "prediction_valid": [True, True, True, False],
        }
        result = performance(records, .8, range(4))
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["relative_performance"], 62.5)
        self.assertEqual(set(load_metrics([2, 2, 2, 2], 4, 2)), {"cv", "valid_token_count"})
        self.assertEqual(load_metrics([4, 4, 0, 0], 4, 2)["cv"], 1)
        self.assertEqual(seed_summary([1, 2, 3])["std"], 1)
        self.assertEqual(layer_summary([0, 1], "max")["worst"], 1)

    def test_selection_metrics_include_adjusted_overlap_and_resolution(self):
        selected = np.array([[0, 2], [2, 3], [0, 1]])
        q = np.array([[8, 1, 6, 3], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=float)
        values = selection_statistics(selected, q)
        self.assertAlmostEqual(values["overlap"][0], 1)
        self.assertAlmostEqual(values["adjusted_overlap"][0], 1)
        self.assertAlmostEqual(values["coverage"][0], 14 / 18)
        self.assertTrue(values["boundary_tie"][1])
        self.assertTrue(values["all_zero"][2])
        self.assertEqual(len(values["normalized_boundary_gap"]), 2)

    def test_churn_and_array_contracts(self):
        left = np.array([[0, 1], [0, 1]])
        right = np.array([[1, 0], [0, 2]])
        values, changed = churn(left, right)
        np.testing.assert_array_equal(values, [0, .5])
        np.testing.assert_array_equal(changed, [0, 1])
        validate_selection(np.array([[0, 255]], dtype=np.uint8), 256, 2)
        validate_labels(np.repeat(np.arange(128), 16).astype(np.int64), 2048, 128, 16)

    def test_e64_import_mapping_and_adjustment(self):
        row = {
            "task": "sst2", "arm": "R4-R2Init", "variant": "default", "k": 13,
            "seed": 0, "role": "best", "group": "oracle_overlap", "layer": "model",
            "metric": "oracle_overlap", "value": .5,
        }
        converted = _convert_row(row)
        self.assertEqual((converted["experts"], converted["arm"]), (64, "R4d"))
        rows = _add_adjusted_overlap([converted])
        adjusted = rows[1]
        self.assertEqual(adjusted["group"], "adjusted_overlap")
        self.assertAlmostEqual(adjusted["value"], (.5 - 13 / 64) / (1 - 13 / 64))

    def test_best_tie_uses_earliest_step(self):
        candidates = [
            {"count": 10, "correct": 8, "state": {"step": 20}},
            {"count": 10, "correct": 8, "state": {"step": 10}},
        ]
        self.assertEqual(choose_best(candidates)["state"]["step"], 10)

    def test_paired_g1_gaps_and_e64_changes_cover_the_protocol(self):
        config = load_config()
        rows = []
        budgets = {
            64: (6, 13, 19, 26),
            128: (13, 26, 38, 51),
            256: (26, 51, 77, 102),
        }
        for task in config["suite"]["tasks"]:
            for experts, values in budgets.items():
                for k in values:
                    for role in ("best", "final"):
                        for metric in ("accuracy", "relative_performance"):
                            for seed in config["suite"]["seeds"]:
                                baseline = .8 if metric == "accuracy" else 100.0
                                for index, arm in enumerate(("G1", "R4o", "R4d", "G2-0.001", "G4")):
                                    rows.append({
                                        "task": task, "experts": experts, "k": k, "arm": arm,
                                        "seed": seed, "role": role, "group": "performance",
                                        "layer": "model", "metric": metric,
                                        "value": baseline + index * .001 * experts / 64,
                                    })
        paired = paired_differences(rows, config)
        changes = gap_changes(paired, config)
        self.assertEqual(len(paired), 384)
        self.assertEqual(len(changes), 256)
        target = next(row for row in changes if row["experts"] == 256 and row["comparison"] == "G4")
        self.assertIsNotNone(target["mean"])
