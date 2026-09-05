import unittest
from unittest.mock import patch

from train import parse_args


class TrainDefaultsTest(unittest.TestCase):
    def parse(self, *args):
        with patch("sys.argv", ["train.py", *args]):
            return parse_args()

    def test_requested_defaults(self):
        args = self.parse()
        self.assertEqual((args.dataset, args.class_start, args.class_count), ("imagefolder", 0, 20))
        self.assertTrue(args.common_classes_only)
        self.assertEqual((args.train_per_class, args.val_per_class), (500, 0))
        self.assertEqual((args.epochs, args.batch_size, args.workers), (100, 32, 4))
        self.assertEqual((args.image_size, args.patch_size, args.prune_at_block), (128, 16, 6))
        self.assertIn("magnitude", args.output_dir)
        self.assertEqual((args.embed_dim, args.num_heads), (256, 4))

    def test_output_paths(self):
        from pathlib import Path
        self.assertIn("tenet_local", self.parse("--pruner", "tenet_local").output_dir)
        self.assertNotEqual(self.parse().output_dir, self.parse().output_dir)
        self.assertEqual(self.parse("--output-dir", "custom_run").output_dir, "custom_run")
        resumed = self.parse("--resume", "saved/last.pth")
        self.assertEqual(Path(resumed.output_dir), Path("saved").resolve())

    def test_full_dataset_override_and_fake(self):
        args = self.parse("--class-count", "0", "--train-per-class", "0", "--val-per-class", "0",
                          "--no-common-classes-only")
        self.assertEqual((args.class_count, args.train_per_class, args.val_per_class), (0, 0, 0))
        self.assertFalse(args.common_classes_only)
        args = self.parse("--dataset", "fake")
        self.assertFalse(args.common_classes_only)
        self.assertEqual((args.class_count, args.train_per_class, args.val_per_class), (0, 0, 0))
