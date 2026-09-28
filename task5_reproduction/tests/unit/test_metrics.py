from unittest import TestCase
import numpy as np

from task5.aggregation.core import layer_summary, seed_summary
from task5.capture.storage import validate_selection
from task5.metrics.load_balance.core import load_metrics
from task5.metrics.performance.core import choose_best, performance
from task5.metrics.selection_quality.core import consistency_summary, expert_pair_matrix, overlap_and_coverage, pair_scores, random_reference
from task5.metrics.stability.core import churn
from task5.substrate.assets import validate_labels


class MetricTests(TestCase):
    def test_accuracy_and_invalid_denominator(self):
        records = {"sample_id": [0, 1, 2, 3], "prediction_right": [True, False, True, False],
                   "prediction_valid": [True, True, True, False]}
        result = performance(records, .8, range(4))
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["relative_performance"], 62.5)
        self.assertEqual(result["invalid_rate"], .25)

    def test_accuracy_rejects_duplicate_or_missing_samples(self):
        records = {"sample_id": [0, 0], "prediction_right": [True, True], "prediction_valid": [True, True]}
        with self.assertRaises(ValueError):
            performance(records)
        records["sample_id"] = [0, 1]
        with self.assertRaises(ValueError):
            performance(records, expected_ids=[0, 1, 2])

    def test_invalid_cannot_be_correct(self):
        with self.assertRaises(ValueError):
            performance({"sample_id": [0], "prediction_right": [True], "prediction_valid": [False]})

    def test_zero_dense_denominator_is_undefined(self):
        self.assertIsNone(performance({"sample_id": [0], "prediction_right": [False], "prediction_valid": [True]}, 0)["relative_performance"])

    def test_best_uses_unrounded_correct_count_and_earliest_step(self):
        candidates = [{"count": 10000, "correct": n, "state": {"step": step}} for n, step in ((9000, 0), (9001, 30), (9001, 20))]
        self.assertEqual(choose_best(candidates)["state"]["step"], 20)
        candidates[0]["correct"] = 9001
        self.assertEqual(choose_best(candidates)["state"]["step"], 0)

    def test_best_requires_equal_population(self):
        with self.assertRaises(ValueError):
            choose_best([{"count": 1}, {"count": 2}])

    def test_uniform_load(self):
        result = load_metrics([2, 2, 2, 2], 4, 2)
        self.assertEqual(result["cv"], 0)
        self.assertEqual(result["gini"], 0)
        self.assertEqual(result["maximum_share"], .25)

    def test_topk_concentrated_load(self):
        result = load_metrics([4, 4, 0, 0], 4, 2)
        self.assertEqual(result["cv"], 1)
        self.assertEqual(result["gini"], .5)
        self.assertEqual(result["maximum_share"], .5)
        with self.assertRaises(ValueError):
            load_metrics([8, 0, 0, 0], 4, 2)

    def test_churn_ignores_order_but_exact_change_detects_one_swap(self):
        a = np.array([[0, 1], [0, 1]])
        b = np.array([[1, 0], [0, 2]])
        values, changed = churn(a, b)
        np.testing.assert_array_equal(values, [0, .5])
        np.testing.assert_array_equal(changed, [0, 1])

    def test_coverage_document_example(self):
        overlap, coverage, zeros = overlap_and_coverage([[0, 3]], [[8, 1, 6, 3]])
        self.assertEqual(overlap[0], .5)
        self.assertAlmostEqual(coverage[0], 11 / 18)
        self.assertEqual(zeros, 0)

    def test_coverage_averages_per_token_and_skips_only_exact_zero(self):
        overlap, coverage, zeros = overlap_and_coverage([[0], [0], [0]], [[1, 0], [0, 9], [0, 0]])
        self.assertEqual(coverage.mean(), .5)
        self.assertEqual(zeros, 1)
        _, small, zeros = overlap_and_coverage([[0]], [[1e-30, 1e-30]])
        self.assertEqual(small[0], .5)
        self.assertEqual(zeros, 0)

    def test_oracle_ties_use_expert_id(self):
        overlap, _, _ = overlap_and_coverage([[0, 1], [2, 3]], [[1, 1, 1, 1], [0, 0, 0, 0]])
        np.testing.assert_array_equal(overlap, [1, 0])

    def test_coactivation_divides_by_n_and_uses_cross_expert_pairs(self):
        activation = np.array([[1, 2, 0, 1], [0, 1, 1, 0]], dtype=np.float32)
        matrix = expert_pair_matrix(activation.T @ activation, 2, [0, 0, 1, 1], 2)
        self.assertEqual(matrix[0, 1], .5)
        self.assertEqual(pair_scores([[0, 1]], matrix)[0], .5)

    def test_coactivation_noncontiguous_labels(self):
        c = np.arange(16).reshape(4, 4).astype(float)
        c += c.T
        q = expert_pair_matrix(c, 2, [1, 0, 1, 0], 2)
        self.assertEqual(q[0, 1], c[np.ix_([1, 3], [0, 2])].mean() / 2)

    def test_random_reference_chunk_invariance(self):
        matrix = np.arange(64, dtype=np.float64).reshape(8, 8)
        matrix += matrix.T
        a = random_reference(matrix, 37, 3, 8, 0, "same", 4)
        b = random_reference(matrix, 37, 3, 8, 0, "same", 19)
        np.testing.assert_array_equal(a, b)

    def test_random_reference_uses_all_tokens_and_uniform_pairs(self):
        matrix = np.ones((4, 4)) * 7
        np.fill_diagonal(matrix, 999)
        np.testing.assert_array_equal(random_reference(matrix, 19, 2, 5, 0, "constant"), [7] * 5)

    def test_random_quantiles_and_zero_ratio(self):
        result = consistency_summary(10, [1, 2, 3, 4])
        self.assertEqual(result["random_mean"], 2.5)
        self.assertEqual(result["ratio"], 4)
        self.assertAlmostEqual(result["random_low"], 1.075)
        self.assertAlmostEqual(result["random_high"], 3.925)
        self.assertIsNone(consistency_summary(1, [0, 0])["ratio"])

    def test_layer_macro_is_unweighted_and_na_is_not_dropped(self):
        summary = layer_summary([0, 1], "min")
        self.assertEqual(summary, {"mean": .5, "layer_std": .5, "worst": 0})
        self.assertIsNone(layer_summary([None, 1])["mean"])

    def test_seed_sample_std_and_single_seed_smoke(self):
        self.assertEqual(seed_summary([1, 2, 3])["std"], 1)
        self.assertIsNone(seed_summary([1], deterministic=True)["std"])
        self.assertIsNone(seed_summary([1])["std"])

    def test_index_and_split_validation(self):
        validate_selection(np.array([[0, 3]], dtype=np.uint8), 4, 2)
        for bad in (np.array([[0, 0]]), np.array([[0, 4]]), np.array([[0., 1.]])):
            with self.assertRaises(ValueError):
                validate_selection(bad, 4, 2)
        validate_labels(np.array([0, 1, 0, 1]), 4, 2, 2)
        with self.assertRaises(ValueError):
            validate_labels(np.array([0., 1., 0., 1.]), 4, 2, 2)
