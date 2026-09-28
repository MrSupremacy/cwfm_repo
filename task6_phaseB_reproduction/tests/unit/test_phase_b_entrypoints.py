from unittest import TestCase

from task6_phaseb.common.config import repository_root


class PhaseBEntrypointTests(TestCase):
    def test_prepare_centroid_audit_is_cpu_exact(self):
        text = (repository_root() / "src/task6_phaseb/common/context.py").read_text(encoding="utf-8")
        self.assertIn("module.wi.weight.detach().cpu()", text)
        self.assertIn('torch.as_tensor(labels[key], device="cpu")', text)
        self.assertIn("np.array_equal(value, artifact_centroids[key])", text)

    def test_stage_wrappers_are_lf_and_exact(self):
        root = repository_root()
        expected = {
            "00_preflight": "preflight", "10_prepare": "prepare", "20_validate": "validate",
            "30_train": "train", "40_capture": "capture", "50_metrics": "metrics",
            "55_import_e64": "import-e64", "60_aggregate": "aggregate",
            "70_tables": "tables", "80_figures": "figures",
        }
        for directory, command in expected.items():
            data = (root / "scripts" / directory / "run.sh").read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertIn(f'run.sh" {command} "$@"', data.decode())

    def test_repository_has_no_old_phase_a_entrypoints(self):
        root = repository_root()
        checked = [
            root / "src/task6_phaseb/cli.py",
            root / "src/task6_phaseb/capture/runner.py",
            root / "src/task6_phaseb/visualization/render.py",
            *list((root / "configs").rglob("*.yaml")),
            *list((root / "scripts").rglob("*.sh")),
        ]
        code = "\n".join(path.read_text(encoding="utf-8") for path in checked).lower()
        for forbidden in ("coactivation", "maximum_share", '"gini"', "r2-soft", "r4-hard", "g3"):
            self.assertNotIn(forbidden, code)

    def test_formal_launcher_is_eight_gpu_and_phase_b_paths(self):
        root = repository_root()
        text = (root / "scripts/90_formal/run_single_node_8gpu.sh").read_text(encoding="utf-8")
        self.assertIn("SHARDS=8", text)
        self.assertIn("phase_b_f0.yaml", text)
        self.assertIn("Expected 2640 checkpoints", text)
        self.assertIn("Expected 2656 A captures", text)
        self.assertIn("matching non-zero PNG/PDF figure counts", text)
        self.assertIn("E64 figures are out of scope", text)
        self.assertNotIn("Expected 48 PNG", text)
        self.assertNotIn("TASK5_", text)
