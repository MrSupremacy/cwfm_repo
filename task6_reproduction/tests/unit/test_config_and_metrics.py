from pathlib import Path
import unittest

import numpy as np

from task6.common.config import ARMS, DENSE_FT_ARM, METRICS, conditions, load_config
from task6.comparison.dense_fullft import _validate_sparse
from task6.metrics.load_balance.core import load_metrics
from task6.metrics.selection_quality.core import overlap_and_coverage
from task6.metrics.stability.core import churn
from task6.visualization.render import dense_fullft_reference


class ConfigAndMetricTests(unittest.TestCase):
    def test_formal_matrix_is_exact(self):
        config = load_config()
        matrix = conditions(config)
        self.assertEqual(len(matrix), 200)
        trained = [condition for condition in matrix if condition.trainable]
        self.assertEqual(len(trained), 198)
        self.assertEqual({condition.arm for condition in trained}, {*ARMS, DENSE_FT_ARM})
        dense_ft = [condition for condition in trained if condition.arm == DENSE_FT_ARM]
        self.assertEqual(len(dense_ft), 6)
        self.assertEqual({condition.k for condition in dense_ft}, {0})
        self.assertEqual({condition.seed for condition in dense_ft}, {0, 1, 2})
        self.assertEqual(tuple(config["metrics"]["enabled"]), METRICS)
        self.assertEqual(config["metrics"]["load_balance"], ["cv"])

    def test_local_paths_resolve_to_task5_inputs(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(local=root / "configs/local/local.yaml")
        self.assertTrue(str((root / config["assets"]["sst2"]["dense"]).resolve()).endswith(
            str(Path("task5/task5_reproduction/inputs/dense/sst2"))))

    def test_cv_only_and_population_definition(self):
        result = load_metrics(np.array([2, 2, 0, 0]), tokens=2, k=2)
        self.assertEqual(set(result), {"cv", "valid_token_count"})
        self.assertAlmostEqual(result["cv"], 1.0)

    def test_overlap_coverage_and_churn(self):
        selected = np.array([[0, 2], [1, 3]])
        q = np.array([[4, 1, 3, 2], [0, 5, 1, 4]], dtype=float)
        overlap, coverage, zeros = overlap_and_coverage(selected, q)
        np.testing.assert_allclose(overlap, [1.0, 1.0])
        np.testing.assert_allclose(coverage, [0.7, 0.9])
        self.assertEqual(zeros, 0)
        value, changed = churn(selected, np.array([[2, 0], [1, 2]]))
        np.testing.assert_allclose(value, [0.0, 0.5])
        np.testing.assert_allclose(changed, [0.0, 1.0])

    def test_relative_plot_uses_task_specific_dense_fullft_best(self):
        row = {
            "task": "sst2", "arm": "dense-ft", "role": "best", "layer": "model",
            "group": "performance", "metric": "relative_performance", "mean": 101.25,
            "std": 0.2,
        }
        self.assertIs(dense_fullft_reference([row], "sst2"), row)
        with self.assertRaises(ValueError):
            dense_fullft_reference([row], "mnli")

    def test_dense_extension_requires_complete_routed_grid_and_init_reference(self):
        config = load_config()
        rows = [{
            "task": task, "arm": variant["arm"], "variant": variant["name"], "k": k,
            "role": "best", "layer": "model", "group": "performance", "metric": "accuracy",
            "mean": 0.5,
        } for task in config["suite"]["tasks"] for variant in config["variants"]
                for k in config["suite"]["top_k"]]
        rows.extend({
            "task": task, "arm": "dense", "variant": "default", "k": 0,
            "role": "static", "layer": "model", "group": "performance",
            "metric": "relative_performance", "mean": 100.0,
        } for task in config["suite"]["tasks"])
        _validate_sparse(rows, config)
        with self.assertRaises(ValueError):
            _validate_sparse(rows[:-1], config)


if __name__ == "__main__":
    unittest.main()
