import os
import tempfile
import unittest
from pathlib import Path

from checkpoints import resolve_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_latest_explicit_directory_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No last.pth"):
                resolve_checkpoint(runs_dir=root)
            for index in (1, 2):
                path = root / str(index) / "last.pth"
                path.parent.mkdir()
                path.touch()
                os.utime(path, (index, index))
            latest = (root / "2" / "last.pth").resolve()
            self.assertEqual(resolve_checkpoint(runs_dir=root), latest)
            self.assertEqual(resolve_checkpoint("latest", root), latest)
            self.assertEqual(resolve_checkpoint(str(root / "1"), root), (root / "1" / "last.pth").resolve())
            self.assertEqual(resolve_checkpoint(str(latest), root), latest)
            with self.assertRaisesRegex(FileNotFoundError, "Available last.pth"):
                resolve_checkpoint(str(root / "single_layer" / "last.pth"), root)
