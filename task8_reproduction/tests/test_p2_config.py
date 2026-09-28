from copy import deepcopy
import unittest

from task8_p01.common.config import load_config as p1_config
from task8_p2.config import best_snapshots, legacy_config, load_config, matrix, validate_run_id


class P2ConfigTests(unittest.TestCase):
    def test_best_only_d9_and_separate_output(self):
        config = load_config()
        self.assertEqual(len(best_snapshots()),27)
        self.assertTrue(all(s.role=="best" for s in best_snapshots()))
        self.assertEqual(matrix()["new_full_evaluations"],81)
        self.assertEqual(matrix()["reused_p1_endpoints"],54)
        self.assertEqual(config["p2"]["multipliers"],[1,2,4,8])
        self.assertEqual(legacy_config(config),p1_config())

    def test_run_id_cannot_escape_output_root(self):
        for value in ("../p1", "a/b", "a\\b", "..", "", "/tmp/job"):
            with self.assertRaises(ValueError):
                validate_run_id(value)
        self.assertEqual(validate_run_id("p2_n_forward01"),"p2_n_forward01")
