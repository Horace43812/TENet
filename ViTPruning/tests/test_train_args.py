import unittest
from unittest.mock import patch

from train import parse_args


class TrainDefaultsTest(unittest.TestCase):
    def parse(self, *args):
        with patch("sys.argv", ["train.py", *args]):
            return parse_args()

    def test_requested_defaults(self):
        args = self.parse()
        self.assertEqual((args.dataset, args.class_start, args.class_count), ("imagefolder", 0, 10))
        self.assertTrue(args.common_classes_only)
        self.assertEqual((args.train_per_class, args.val_per_class), (100, 20))
        self.assertEqual((args.epochs, args.batch_size, args.workers), (50, 32, 4))
        self.assertEqual((args.image_size, args.patch_size, args.prune_at_block), (64, 8, 6))
        self.assertEqual(args.output_dir, "checkpoints/subset_common_magnitude")

    def test_full_dataset_override_and_fake(self):
        args = self.parse("--class-count", "0", "--train-per-class", "0", "--val-per-class", "0",
                          "--no-common-classes-only")
        self.assertEqual((args.class_count, args.train_per_class, args.val_per_class), (0, 0, 0))
        self.assertFalse(args.common_classes_only)
        args = self.parse("--dataset", "fake")
        self.assertFalse(args.common_classes_only)
        self.assertEqual((args.class_count, args.train_per_class, args.val_per_class), (0, 0, 0))
