from unittest import TestCase

from task6_phased.aggregation.pipeline import summarize


class AggregationTests(TestCase):
    def test_best_rows_are_not_split_by_best_equals_final_status(self):
        rows = []
        for seed, value, best_is_final in ((0, .7, True), (1, .8, False), (2, .9, False)):
            rows.append({
                "experts": 128, "arm": "R4d", "variant": "default", "k": 13, "seed": seed,
                "state": f"seed_{seed}_best", "step": 2640 if best_is_final else 2376,
                "group": "specialization", "domain": "structured", "metric": "example",
                "value": value, "is_best": True, "is_final": best_is_final,
            })

        best = [row for row in summarize(rows) if row["role"] == "best"]

        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["n"], 3)
        self.assertAlmostEqual(best[0]["mean"], .8)
        self.assertAlmostEqual(best[0]["std"], .1)
        self.assertTrue(best[0]["is_best"])
        self.assertFalse(best[0]["is_final"])

    def test_expert_counts_are_separate_aggregation_units(self):
        rows = []
        for experts, value in ((128, .8), (256, .9)):
            rows.append({
                "experts": experts, "arm": "R2", "variant": "default",
                "k": 13 if experts == 128 else 26, "seed": None,
                "state": "static", "step": 0, "group": "performance",
                "domain": "sst2", "metric": "native", "value": value,
                "is_best": True, "is_final": True,
            })
        best = [row for row in summarize(rows) if row["role"] == "best"]
        self.assertEqual({row["experts"] for row in best}, {128, 256})
        self.assertTrue(all(row["n"] == 1 for row in best))
