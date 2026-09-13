"""Compare interrupted/resumed training against uninterrupted training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import train


class ResumeTests(unittest.TestCase):
    def test_epoch_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(name, resume=False):
                argv = ["train.py", "--smoke", "--cpu", "--epochs", "2", "--output-dir", str(root / name)]
                with patch("sys.argv", argv + (["--resume"] if resume else [])):
                    train.main()
            run("continuous")
            original_epoch = train.epoch
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt()
                return original_epoch(*args, **kwargs)
            with patch.object(train, "epoch", interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run("resumed")
            run("resumed", True)
            continuous = torch.load(root / "continuous/last.pth", weights_only=False)
            resumed = torch.load(root / "resumed/last.pth", weights_only=False)
            self.assertEqual(resumed["epoch"], 2)
            for name, value in continuous["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
            for a, b in zip(continuous["history"], resumed["history"]):
                self.assertEqual(a["train"], b["train"])
                self.assertEqual(a["dev"], b["dev"])


if __name__ == "__main__":
    unittest.main()
