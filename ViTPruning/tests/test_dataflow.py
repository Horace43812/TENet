"""Run from ViTPruning: python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from datasets import build_dataloaders
from model_vit import VisionTransformer
from train import run_epoch


class ImageFolderDataflowTest(unittest.TestCase):
    def test_subset_mapping_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            for split in ("train", "val"):
                for name in ("a", "b", "c", "d"):
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    for index in range(5):
                        Image.new("RGB", (24, 20), (index * 40, 80, 120)).save(folder / f"{index}.JPEG")
            def build():
                return build_dataloaders("imagefolder", root, 16, 2, 0, False, 1000, 8, 4,
                                         class_start=1, class_count=2, train_per_class=3, val_per_class=2, seed=7)
            data, repeat = build(), build()
            self.assertEqual(data.class_names, ["b", "c"])
            self.assertEqual((data.num_classes, data.train_samples, data.eval_samples), (2, 6, 4))
            self.assertEqual(data.train_loader.dataset.class_to_idx, data.eval_loader.dataset.class_to_idx)
            self.assertEqual(data.train_loader.dataset.samples, repeat.train_loader.dataset.samples)
            self.assertEqual(sorted(data.train_loader.dataset.targets), [0, 0, 0, 1, 1, 1])
            model = VisionTransformer(image_size=16, patch_size=4, num_classes=2,
                                       embed_dim=12, depth=2, num_heads=3, pruner_name="magnitude")
            criterion = torch.nn.CrossEntropyLoss()
            metrics = run_epoch(model, data.train_loader, criterion, torch.device("cpu"),
                                torch.optim.AdamW(model.parameters()))
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertIn("accuracy", run_epoch(model, data.eval_loader, criterion, torch.device("cpu")))
            full = build_dataloaders("imagefolder", root, 16, 2, 0, False, 1000, 8, 4)
            self.assertEqual((full.num_classes, full.train_samples, full.eval_samples), (4, 20, 20))
            with self.assertRaisesRegex(ValueError, "at least two"):
                build_dataloaders("imagefolder", root, 16, 2, 0, False, 1000, 8, 4, class_start=4)
            (root / "val" / "extra").mkdir()
            self.assertEqual(build().class_names, ["b", "c"])
            (root / "val" / "a").rename(root / "val" / "renamed_a")
            self.assertEqual(build().class_names, ["b", "c"])
            (root / "val" / "b").rename(root / "val" / "renamed_b")
            with self.assertRaisesRegex(ValueError, "missing 1 selected training classes:.*b"):
                build()


if __name__ == "__main__":
    unittest.main()
