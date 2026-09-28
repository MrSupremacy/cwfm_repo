from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from task6_phaseb.common.config import ARMS
from task6_phaseb.visualization import render


def row(experts, k, arm, metric, *, layer="model", role=None, epoch=None):
    role = role or ("static" if arm == "R2" else "best")
    return {
        "task": "sst2",
        "experts": experts,
        "k": k,
        "arm": arm,
        "variant": "default",
        "group": "performance" if metric in ("accuracy", "relative_performance") else "load_balance",
        "metric": metric,
        "layer": layer,
        "role": role,
        "epoch": epoch,
        "mean": 100.0 if metric == "relative_performance" else 0.5,
        "std": None if arm == "R2" else 0.01,
        "deterministic": arm == "R2",
    }


class VisualizationLayoutTests(unittest.TestCase):
    def payloads(self):
        budgets = {64: 6, 128: 13, 256: 26}
        aggregated = []
        normalized = []
        for experts, k in budgets.items():
            for arm in ARMS:
                aggregated.extend([
                    row(experts, k, arm, "relative_performance"),
                    row(experts, k, arm, "cv"),
                ])
                if experts in (128, 256):
                    normalized.append(row(experts, k, arm, "cv", layer="encoder_layer_00"))
                    if arm != "R2":
                        aggregated.append(
                            row(experts, k, arm, "relative_performance", role="trajectory", epoch=0)
                        )
        aggregated.append({
            **row(0, 0, "R2", "accuracy"),
            "arm": "dense",
            "role": "static",
            "deterministic": True,
            "std": None,
        })
        paired = [
            {
                "task": "sst2", "experts": experts, "k": k, "role": "best",
                "reference": "G1", "comparison": "G4", "metric": "relative_performance",
                "unit": "relative_percentage_points", "mean": 0.0, "std": 0.0,
            }
            for experts, k in budgets.items()
        ]
        changes = [{
            "task": "sst2", "budget": "ratio10", "experts": 128, "k": 13,
            "baseline_experts": 64, "baseline_k": 6, "role": "best", "reference": "G1",
            "comparison": "G4", "metric": "relative_performance",
            "unit": "relative_percentage_points", "mean": 0.0, "std": 0.0,
        }]
        return {
            "aggregated": {"rows": aggregated},
            "normalized": {"rows": normalized},
            "paired_differences": {"rows": paired},
            "gap_changes": {"rows": changes},
        }

    def test_per_expert_tables_and_table_only_cross_expert_comparison(self):
        payloads = self.payloads()
        config = {"experiment": {"budgets": {128: [13], 256: [26]}}}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            render, "data_file", side_effect=lambda _c, _r, section, **_k: payloads[section]
        ), mock.patch.object(
            render, "_artifact_rows", return_value=[
                {"experts": 128, "complete": True}, {"experts": 256, "complete": True}
            ]
        ), mock.patch.object(
            render, "_resource_rows", return_value=[
                {"experts": 64, "resource": "ok"},
                {"experts": 128, "resource": "ok"},
                {"experts": 256, "resource": "ok"},
            ]
        ):
            root = Path(directory)
            render.tables(config, "layout", result_root=root)
            for experts in (128, 256):
                expert = root / "tables" / f"{experts}_experts"
                self.assertTrue((expert / "main" / "layout" / "best_validation.csv").is_file())
                self.assertTrue((expert / "diagnostics" / "layout" / "paired_differences_vs_G1.csv").is_file())
                self.assertTrue((expert / "appendix" / "layout" / "per_layer_per_seed.csv").is_file())
            comparison = root / "tables" / "expert_comparison" / "layout"
            self.assertTrue((comparison / "00_dense_init_reference.csv").is_file())
            self.assertTrue((comparison / "03_gap_changes_vs_E64.csv").is_file())

    def test_figure_paths_reject_e64_and_cross_expert_folders(self):
        root = Path("results/figures")
        self.assertEqual(
            render._figure_path(root, 128, "main", "run01", "plot"),
            root / "128_experts" / "main" / "run01" / "plot",
        )
        self.assertEqual(
            render._figure_path(root, 256, "appendix", "run01", "plot"),
            root / "256_experts" / "appendix" / "run01" / "plot",
        )
        with self.assertRaises(ValueError):
            render._figure_path(root, 64, "main", "run01", "plot")
        with self.assertRaises(ValueError):
            render._figure_path(root, 128, "expert_comparison", "run01", "plot")


if __name__ == "__main__":
    unittest.main()
