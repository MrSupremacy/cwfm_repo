import tempfile
import gzip
import csv
from pathlib import Path
from unittest import TestCase


class IOTests(TestCase):
    def test_completion_marker_detects_mutation(self):
        from task8_p01.common.io import checked_complete, complete

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "value.txt").write_text("first", encoding="utf-8")
            complete(root, {"id": 8})
            self.assertEqual(checked_complete(root), {"id": 8})
            (root / "value.txt").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_complete(root)

    def test_completion_marker_after_gzip_finalize(self):
        from task8_p01.common.io import checked_complete, complete, fresh_directory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "replay"
            with fresh_directory(root), gzip.open(
                root / "token_records.jsonl.gz", "wt", encoding="utf-8"
            ) as stream:
                stream.write('{"value": 1}\n')
                stream.close()
                complete(root, {"kind": "replay"})
            self.assertEqual(checked_complete(root), {"kind": "replay"})

    def test_write_csv_accepts_optional_metric_fields(self):
        from task8_p01.common.io import write_csv

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metrics.csv"
            write_csv(output, [
                {"task": "sst2", "accuracy": 0.9},
                {"task": "qqp", "accuracy": 0.8, "f1": 0.7},
            ])
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["f1"], "")
            self.assertEqual(rows[1]["f1"], "0.7")
