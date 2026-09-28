from copy import deepcopy
from unittest import TestCase

import numpy as np

from task6_phased.capture.storage import validate_selection
from task6_phased.common.config import (
    ARMS, Condition, budgets_for, conditions, expected_matrix_counts, load_config,
    resolved_split, validate_config,
)
from task6_phased.dense.evaluation import native_metrics
from task6_phased.metrics.load_balance.core import load_metrics
from task6_phased.metrics.selection_quality.core import selection_statistics
from task6_phased.metrics.specialization.core import domain_metrics, js_distance
from task6_phased.metrics.stability.core import churn


class PhaseDProtocolTests(TestCase):
    def test_formal_matrix(self):
        config = load_config()
        self.assertEqual(expected_matrix_counts(config), {
            "conditions": 96, "training_runs": 90, "static_states": 6,
            "router_checkpoints": 990, "routed_states": 996,
        })
        items = conditions(config)
        self.assertEqual({item.experts for item in items}, {128, 256})
        self.assertEqual(budgets_for(config, 128), (13, 19, 26))
        self.assertEqual(budgets_for(config, 256), (26, 38, 51))
        self.assertEqual(tuple(item["arm"] for item in config["variants"]), ARMS)
        self.assertTrue(all(item.seed is None for item in items if item.arm == "R2"))

    def test_condition_path_has_no_task_dimension(self):
        value = Condition(64, "G4", 10, 2)
        self.assertEqual(value.path.as_posix(), "E_64/G4/default/k_10/seed_2")

    def test_split_seed_and_version_are_frozen(self):
        config = load_config()
        self.assertEqual(config["split_generation"]["random_state"], 1)
        self.assertEqual(config["split_generation"]["k_means_constrained_version"], "0.9.1")
        changed = deepcopy(config)
        changed["split_generation"]["random_state"] = 0
        with self.assertRaises(ValueError):
            validate_config(changed)
        self.assertEqual(
            (resolved_split(config, 128)["n_clusters"], resolved_split(config, 128)["size_min"]),
            (128, 16),
        )
        self.assertEqual(
            (resolved_split(config, 256)["n_clusters"], resolved_split(config, 256)["size_min"]),
            (256, 8),
        )

    def test_only_cv_load_metric(self):
        self.assertEqual(load_metrics([2, 2, 2, 2], 4, 2)["cv"], 0)
        self.assertEqual(load_metrics([4, 4, 0, 0], 4, 2)["cv"], 1)

    def test_native_metrics(self):
        qqp = native_metrics("qqp", [1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(qqp["accuracy"], .5)
        self.assertAlmostEqual(qqp["f1"], .5)
        self.assertAlmostEqual(qqp["native"], .5)
        qnli = native_metrics("qnli", [1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(qnli["accuracy"], .5)
        self.assertEqual(qnli["native"], .5)

    def test_selection_churn_and_specialization(self):
        selected = np.array([[0, 2], [2, 3]])
        q = np.array([[8, 1, 6, 3], [1, 1, 1, 1]], dtype=float)
        values = selection_statistics(selected, q)
        self.assertAlmostEqual(values["overlap"][0], 1)
        self.assertAlmostEqual(values["coverage"][0], 14 / 18)
        churn_value, _ = churn(selected, np.array([[2, 0], [2, 1]]))
        np.testing.assert_allclose(churn_value, [0, .5])
        validate_selection(selected, 4, 2)
        same = np.array([.25, .25, .25, .25])
        self.assertEqual(js_distance(same, same), 0)
        specialization = domain_metrics({
            "sst2": [1, 0], "mnli": [0, 1], "qnli": [1, 0], "qqp": [0, 1],
        }, ("sst2", "mnli", "qnli", "qqp"))
        self.assertAlmostEqual(specialization["mi_bits"], 1)
        self.assertAlmostEqual(specialization["nmi_domain"], .5)

    def test_forbidden_metric_rejected(self):
        config = load_config()
        changed = deepcopy(config)
        changed["metrics"]["extra"] = ["coactivation"]
        with self.assertRaises(ValueError):
            validate_config(changed)
