import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from task5.common.config import repository_root


class ShellEntrypointTests(TestCase):
    def test_stage_launchers_are_thin_lf_shell_wrappers(self):
        root = repository_root()
        expected = {
            "00_preflight": "preflight", "10_prepare": "prepare", "20_validate": "validate",
            "30_train": "train", "40_capture": "capture", "50_metrics": "metrics",
            "60_aggregate": "aggregate", "70_tables": "tables", "80_figures": "figures",
        }
        self.assertFalse(list((root / "scripts").glob("*/run.py")))
        for directory, command in expected.items():
            data = (root / "scripts" / directory / "run.sh").read_bytes()
            self.assertNotIn(b"\r", data)
            text = data.decode()
            self.assertIn("set -euo pipefail", text)
            self.assertIn(f"run.sh\" {command} \"$@\"", text)

    @skipUnless(os.name == "posix", "Linux Bash execution regression")
    def test_launcher_finds_repository_from_another_directory(self):
        root = repository_root()
        env = {**os.environ, "PYTHON": sys.executable}
        with TemporaryDirectory() as work:
            result = subprocess.run(
                ["bash", str(root / "scripts/00_preflight/run.sh"), "--config-only"],
                cwd=work, env=env, capture_output=True, text=True, check=True)
        parsed = json.loads(result.stdout)
        self.assertEqual((parsed["conditions"], parsed["training_runs"]), (242, 192))

    @skipUnless(os.name == "posix", "Linux Bash execution regression")
    def test_launcher_preserves_argument_boundaries_and_exit_status(self):
        root = repository_root()
        env = {**os.environ, "PYTHON": sys.executable}
        result = subprocess.run(
            ["bash", str(root / "scripts/00_preflight/run.sh"), "--config-only", "--run-id", "bad id"],
            env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run-id", result.stderr)
