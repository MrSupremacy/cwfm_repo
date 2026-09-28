"""Synthetic data only; never presented as experimental measurements."""
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import importlib.util

from task5.aggregation.phase_a import BASE_METHODS, METHODS, report, result_root, validate_rows
from task5.aggregation.pipeline import collect_rows
from task5.common.config import conditions, root_for
from task5.common.io import read_json, sha256, write_json
from task5.metrics.performance.pipeline import select_best
from task5.metrics.pipeline import compute_condition
from tests.fixtures.offline import build_fixture, fixture_config, mock_probe_chunks, mock_table_read


class PhaseAReportTest(TestCase):
    def test_supplement_metrics_complete_and_report_does_not_touch_old_results(self):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            c = fixture_config(temp, phase_a=True)
            run = "fixture_phase_a"
            build_fixture(c, run)
            stack.enter_context(patch("task5.metrics.performance.pipeline.read_parquet", side_effect=mock_table_read))
            stack.enter_context(patch("task5.metrics.pipeline.read_parquet", side_effect=mock_table_read))
            stack.enter_context(patch("task5.metrics.pipeline.probe_chunks", side_effect=mock_probe_chunks))
            for condition in conditions(c):
                select_best(c, condition, run)
                compute_condition(c, condition, run)
            rows = [r for r in collect_rows(c, run) if r["arm"] != "dense"]
            validate_rows(rows, c, METHODS)
            for missing in (lambda r: r["arm"] == "R2-soft",
                            lambda r: r["arm"] == "R4-hard" and r["seed"] == 2,
                            lambda r: r["arm"] == "R4-hard" and r["group"] == "activation_coverage" and r["epoch"] == 3):
                with self.assertRaises(ValueError):
                    validate_rows([r for r in rows if not missing(r)], c, METHODS)
            baseline = root_for(c) / "results/data/normalized" / run / "metrics.json"
            base_rows = [r for r in rows if (r["arm"], r["variant"]) in BASE_METHODS]
            write_json(baseline, {"rows": base_rows, "fixture_only": True})
            before = sha256(baseline)
            stack.enter_context(patch("task5.aggregation.phase_a.check_base", return_value=(base_rows, {"fixture_only": True})))
            if importlib.util.find_spec("matplotlib"):
                stack.enter_context(patch("task5.visualization.render.DISPLAY", ["relative_performance"]))
            else:
                stack.enter_context(patch("task5.visualization.render.figures"))
            report(c, run)
            self.assertEqual(sha256(baseline), before)
            target = result_root(c)
            data = read_json(target / "data/aggregated" / run / "metrics.json")["rows"]
            self.assertEqual({(r["arm"], r["variant"]) for r in data}, METHODS)
            self.assertFalse((root_for(c) / "results/figures").exists())
            static = [r for r in data if r["arm"] == "R2-soft"]
            self.assertTrue(all(r["n"] == 1 and r["deterministic"] for r in static))
            self.assertFalse(any(r["group"] == "churn" for r in static))
            self.assertEqual({r["epoch"] for r in data if r["arm"] == "R4-hard" and r["role"] == "trajectory"
                              and r["metric"] == "activation_coverage"}, set(range(11)))
            self.assertTrue((target / "tables/main" / run / "best_validation.csv").is_file())
