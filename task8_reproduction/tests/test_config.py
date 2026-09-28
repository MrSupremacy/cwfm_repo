import importlib.util
from unittest import TestCase, skipUnless


@skipUnless(importlib.util.find_spec("yaml"), "PyYAML unavailable")
class ConfigTests(TestCase):
    def test_frozen_matrix(self):
        from task8_p01.common.config import cells, load_config, matrix_counts, snapshots

        config = load_config()
        self.assertEqual(config["protocol"]["checkpoint_roles"], ["init", "best"])
        self.assertEqual(len(cells()), 9)
        self.assertEqual(len(snapshots()), 36)
        self.assertEqual(matrix_counts()["training_runs"], 0)
        self.assertEqual(matrix_counts()["four_cell_rows_per_population"], 117)

    def test_init_is_seed_deduplicated(self):
        from task8_p01.common.config import Snapshot

        self.assertIsNone(Snapshot(64, 6, None, "init").seed)
        with self.assertRaises(ValueError):
            Snapshot(64, 6, 0, "init")
