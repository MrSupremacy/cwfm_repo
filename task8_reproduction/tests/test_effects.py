import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("yaml"), "PyYAML unavailable")
class EffectTests(TestCase):
    def test_factorial_equations(self):
        from task8_p01.reporting.aggregate import factorial_effects

        rows = []
        for mode, value, seed, role in (
            ("M00", 1.0, None, "static"), ("M01", 1.1, 0, "best"),
            ("M10", 1.2, 0, "best"), ("M11", 1.5, 0, "best"),
            ("M01", .8, None, "init"), ("M10", 1.0, None, "init"),
            ("M11", .8, None, "init"),
        ):
            rows.append({
                "E": 64, "k": 6, "seed": seed, "checkpoint_role": role,
                "task": "sst2", "mode": mode, "native": value,
            })
        result = factorial_effects(rows)
        self.assertEqual(len(result), 2)
        by_role = {row["checkpoint_role"]: row for row in result}
        self.assertAlmostEqual(by_role["best"]["aggregation_R2"], .1)
        self.assertAlmostEqual(by_role["best"]["aggregation_R4"], .3)
        self.assertAlmostEqual(by_role["best"]["interaction"], .2)
        self.assertAlmostEqual(by_role["init"]["aggregation_R2"], -.2)
        self.assertAlmostEqual(by_role["init"]["selection_uniform"], 0)
