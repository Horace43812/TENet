import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from graph import group_windows
from model import TENet
from analyze import Descriptor, split_validation, pair_changes


class MechanismTest(unittest.TestCase):
    def test_window_layout_and_mean(self):
        x = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6).float()
        grouped = group_windows(x)
        expected = torch.stack([x[b, :, y:y+2, z:z+2].reshape(3, 4)
            for b in range(2) for y in range(0, 4, 2) for z in range(0, 6, 2)])
        torch.testing.assert_close(grouped, expected)
        restored = grouped.mean(-1).reshape(2, 2, 3, 3).permute(0, 3, 1, 2)
        torch.testing.assert_close(restored, torch.nn.functional.avg_pool2d(x, 2))

    def test_observers_and_gradient(self):
        torch.manual_seed(1)
        model = TENet(3, image_size=8, layers=2, alpha=2, head_grid=1)
        model.eval()
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        baseline = model(x)
        captured = {}
        model.layers[0].observer = lambda name, value: captured.update({name: value})
        torch.testing.assert_close(model(x), baseline)
        self.assertEqual(captured["vertices"].shape, (2, 12, 4, 4))
        self.assertEqual(captured["graph_raw"].shape, (2, 300, 4, 4))
        self.assertEqual(captured["output"].shape, (2, 5, 4, 4))
        # Inspect the first physical window, preserving channel and matrix coordinates.
        layer = model.layers[0]
        nodes = captured["nodes"][0, :, 0, 0].reshape(3, 10)
        expected_graph = nodes.unsqueeze(2) * layer.L * nodes.unsqueeze(1)
        torch.testing.assert_close(captured["graph_raw"][0, :, 0, 0].reshape(3, 10, 10), expected_graph)
        torch.testing.assert_close(captured["trace"][0, :, 0, 0], expected_graph.diagonal(dim1=-2, dim2=-1).sum(-1))
        Descriptor(1, 8, 42)(captured["output"], detach=False).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for mode in ["no_trace", "no_cross", "no_graph_norm"]:
            model.layers[0].intervention = mode
            self.assertTrue(torch.isfinite(model(x)).all())

    def test_splits_and_pair_errors(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        discovery, confirmation = split_validation(labels, 42)
        self.assertFalse(set(discovery.tolist()) & set(confirmation.tolist()))
        self.assertEqual(len(discovery) + len(confirmation), len(labels))
        pairs = pair_changes(labels, labels, 1 - labels, discovery, 2)
        self.assertEqual(pairs[(0, 1)]["delta_pp"], 100)

    def test_train_analyze_imagefolder(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ["train", "val"]:
                for name, color in [("a", (200, 20, 10)), ("b", (10, 20, 200))]:
                    folder = root / "dataset" / split / name
                    folder.mkdir(parents=True)
                    for i in range(4):
                        Image.new("RGB", (12, 12), color).save(folder / f"{i}.JPEG")
            run = root / "run"
            subprocess.run([sys.executable, "train.py", "--data-dir", str(root / "dataset"),
                "--output-dir", str(run), "--epochs", "1", "--image-size", "8", "--head-grid", "1",
                "--layers", "1", "--alpha", "1", "--batch-size", "2", "--workers", "0", "--cpu"], cwd=workspace, check=True)
            analysis = root / "analysis"
            subprocess.run([sys.executable, "analyze.py", "--checkpoint", str(run / "last.pth"),
                "--output-dir", str(analysis), "--probe-epochs", "1", "--grid", "1", "--probe-dim", "8",
                "--batch-size", "2", "--intervention", "no_trace", "--cpu"], cwd=workspace, check=True)
            report = json.loads((analysis / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["class_names"], ["a", "b"])
            self.assertIn("output", report["metrics"])
            self.assertEqual(report["intervention"]["name"], "no_trace")
            self.assertTrue((analysis / "class_pairs.csv").exists())
            subprocess.run([sys.executable, "inspect_pair.py", "--analysis-dir", str(analysis),
                "--class-a", "a", "--class-b", "b", "--max-samples", "1"], cwd=workspace, check=True)
            self.assertEqual(len(list(analysis.glob("pair_*/*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
