from pathlib import Path
import tempfile
import unittest

from task8_p01.common.io import checked_complete
from task8_p2.artifacts import atomic_artifact, reuse


class P2ArtifactTests(unittest.TestCase):
    def test_publish_resume_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"job"
            header = {"hash":"registered","role":"best"}
            with atomic_artifact(path,header):
                pass
            self.assertTrue(reuse(path,header))
            with self.assertRaises(ValueError):
                reuse(path,{"hash":"wrong"})
            (path/"identity.json").write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_complete(path)

    def test_failed_work_never_publishes_complete_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"job"
            with self.assertRaises(RuntimeError):
                with atomic_artifact(path,{"job":1}):
                    raise RuntimeError("injected failure")
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(tmp).glob(".job.partial-*"))),1)
