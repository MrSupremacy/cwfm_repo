"""Run available tests; report skipped dependency-bound tests separately."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.metadata
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "tmp/local-tests/report.json"))
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    versions = {}
    for name in ("numpy", "torch", "transformers", "datasets", "pyarrow", "matplotlib", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    from task5.common.config import implementation_id
    from task5.common.io import write_json
    write_json(args.report, {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                            "versions": versions, "implementation": implementation_id(),
                            "total": result.testsRun, "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                            "failures": [str(t) for t, _ in result.failures], "errors": [str(t) for t, _ in result.errors],
                            "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
                            "success": result.wasSuccessful(), "scope": "unit and synthetic fixtures; not real-data CUDA validation"})
    print(f"Test report: {args.report}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
