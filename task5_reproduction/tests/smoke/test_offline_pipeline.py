from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import importlib.util
import io
import sys

from task5.aggregation.pipeline import aggregate
from task5.common.config import Condition, conditions, root_for, run_path
from task5.common.io import read_json
from task5.metrics.performance.pipeline import best_state, select_best
from task5.metrics.pipeline import compute_condition
from task5.visualization.render import figures, tables
from tests.fixtures.offline import build_fixture, fixture_config, mock_probe_chunks, mock_table_read


class OfflinePipelineTest(TestCase):
    def test_all_metrics_best_selection_seed_aggregation_and_tables_without_models(self):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            config = fixture_config(temp)
            run_id = "fixture01"
            build_fixture(config, run_id)
            stack.enter_context(patch("task5.metrics.performance.pipeline.read_parquet", side_effect=mock_table_read))
            stack.enter_context(patch("task5.metrics.pipeline.read_parquet", side_effect=mock_table_read))
            stack.enter_context(patch("task5.metrics.pipeline.probe_chunks", side_effect=mock_probe_chunks))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for condition in conditions(config):
                select_best(config, condition, run_id)
            r4_seed1 = Condition("sst2", "R4", "default", 6, 1)
            self.assertEqual(best_state(config, r4_seed1, run_id)["epoch"], 5)
            self.assertEqual(best_state(config, Condition("sst2", "R4", "default", 6, 0), run_id)["epoch"], 0)
            for condition in conditions(config):
                compute_condition(config, condition, run_id)
            aggregate(config, run_id)
            tables(config, run_id)
            has_matplotlib = bool(importlib.util.find_spec("matplotlib"))
            if has_matplotlib:
                figures(config, run_id)
            output = read_json(root_for(config) / "results/data/aggregated" / run_id / "metrics.json")["rows"]
            selected = [r for r in output if r["arm"] == "R4" and r["role"] == "best" and r["layer"] == "model" and r["metric"] == "accuracy"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["mean"], .75)
            self.assertEqual(selected[0]["n"], 3)
            self.assertEqual([s["epoch"] for s in selected[0]["states"]], [0, 5, 0])
            r2init = [r for r in output if r["arm"] == "R4-R2Init" and r["role"] == "best"
                      and r["layer"] == "model" and r["metric"] == "accuracy"]
            self.assertEqual(len(r2init), 1)
            self.assertEqual(r2init[0]["n"], 3)
            paired = read_json(root_for(config) / "results/data/aggregated" / run_id / "paired_differences.json")["rows"]
            self.assertTrue(any(r["reference"] == "R4" and r["comparison"] == "R4-R2Init" for r in paired))
            trajectory = [r for r in output if r["arm"] == "R4" and r["role"] == "trajectory" and r["layer"] == "model" and r["metric"] == "churn"]
            self.assertEqual([r["epoch"] for r in trajectory], list(range(1, 11)))
            self.assertTrue((root_for(config) / "results/tables/main" / run_id / "best_validation.md").is_file())
            if has_matplotlib:
                self.assertTrue((root_for(config) / "results/figures/main" / run_id /
                                 "sst2_performance_relative_performance.png").is_file())
            self.assertFalse((root_for(config) / "runs/train").exists())
            # Missing data must not silently reduce the experiment or seed population.
            missing = run_path(config, "metrics/performance", r4_seed1, run_id) / "step_5/metrics.json"
            moved = missing.with_suffix(".saved.json")
            missing.rename(moved)
            with self.assertRaises(FileNotFoundError):
                aggregate(config, run_id)
