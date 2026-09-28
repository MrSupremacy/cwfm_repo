from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import hashlib
import importlib.util
import unittest
import numpy as np

from task6_phased.capture.storage import (
    ProbeWriter, probe_chunks, read_parquet, validate_selection, write_parquet,
)
from task6_phased.common.io import checked_complete, complete, fresh_output, write_json


class IOTests(TestCase):
    def test_completion_detects_corruption_and_wrong_identity(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            write_json(path / "payload.json", {"value": 3})
            complete(path, {"id": "first"})
            self.assertEqual(checked_complete(path), {"id": "first"})
            with self.assertRaises(ValueError):
                checked_complete(path, {"id": "second"})
            write_json(path / "payload.json", {"value": 4})
            with self.assertRaises(ValueError):
                checked_complete(path)

    def test_raw_output_is_never_overwritten(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError), fresh_output(Path(temp)):
                pass

    def test_nonfinite_json_rejected(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                write_json(Path(temp) / "bad.json", {"score": float("nan")})

    def test_probe_writer_shards_and_checks_complete_keys(self):
        with TemporaryDirectory() as temp:
            keys = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
            expected = {"count": 3, "sha256": hashlib.sha256(keys.astype("<i8").tobytes()).hexdigest()}
            observed = []

            def sink(path, columns, *args):
                observed.append((path, {k: np.asarray(v).copy() for k, v in columns.items()}))

            with patch("task6_phased.capture.storage.write_parquet", side_effect=sink):
                writer = ProbeWriter(temp, "encoder_layer_00", expected, 2, 4, True, 2, 3)
                writer.add(keys[:1], np.array([[0, 1]]), np.ones((1, 4)))
                writer.add(keys[1:], np.array([[2, 3], [0, 3]]), np.ones((2, 4)))
                writer.finish()
            self.assertEqual([len(v[1]["sample_id"]) for v in observed], [2, 1])
            self.assertEqual(observed[0][1]["selected_experts"].dtype, np.uint8)
            self.assertEqual(observed[0][1]["expert_activation_sums"].dtype, np.float32)

    def test_probe_duplicate_and_missing_rows_fail(self):
        expected = {"count": 2, "sha256": "not-used"}
        with patch("task6_phased.capture.storage.write_parquet"):
            writer = ProbeWriter("unused", "encoder_layer_00", expected, 2, 4, False, 8, 3)
            writer.add(np.array([[0, 0]]), np.array([[0, 1]]))
            with self.assertRaises(ValueError):
                writer.add(np.array([[0, 0]]), np.array([[0, 1]]))
            with self.assertRaises(ValueError):
                writer.finish()

    def test_e256_selected_ids_fit_the_frozen_uint8_contract(self):
        selected = np.array([[0, 255], [127, 254]], dtype=np.uint8)
        validate_selection(selected, 256, 2)
        with self.assertRaises(ValueError):
            validate_selection(np.array([[255, 255]], dtype=np.int64), 256, 2)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is not installed; real Parquet I/O awaits server environment")
class ArrowTests(TestCase):
    def test_real_parquet_schema_and_roundtrip(self):
        import pyarrow.parquet as pq
        import pyarrow as pa
        with TemporaryDirectory() as temp:
            path = Path(temp) / "part_00000.parquet"
            columns = {"sample_id": [3], "token_position": [1], "layer_id": ["encoder_layer_00"],
                       "selected_experts": np.array([[0, 3]], dtype=np.uint8),
                       "expert_activation_sums": np.array([[8, 1, 6, 3]], dtype=np.float32)}
            write_parquet(path, columns, "CD", 4, 2)
            schema = pq.read_schema(path)
            self.assertEqual(schema.field("sample_id").type, pa.int64())
            self.assertEqual(schema.field("selected_experts").type.value_type, pa.uint8())
            self.assertEqual(schema.field("expert_activation_sums").type.value_type, pa.float32())
            np.testing.assert_array_equal(next(probe_chunks(temp))["expert_activation_sums"], columns["expert_activation_sums"])
            self.assertEqual(read_parquet(path)["sample_id"], [3])

