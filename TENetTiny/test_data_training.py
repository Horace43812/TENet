import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from data import build_data
from train import main, parser


class MiniOfficial(Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        self.targets = [0] * 4 + [1] * 4 if train else [0, 1]
        self.classes = ['a', 'b']
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.new('L', (28, 28), color=index)), self.targets[index]


class DataTrainingTests(unittest.TestCase):
    def test_official_train_dev_are_disjoint_and_seed_independent(self):
        args = parser().parse_args(['--dev-per-class', '1'])
        with patch.dict('data.SPECS', {'mnist': (MiniOfficial, 1, 28, (.1307,), (.3081,), 500)}):
            train, dev, test, count, split = build_data(args)
            self.assertEqual((len(train), len(dev), len(test), count), (6, 2, 2, 2))
            self.assertFalse(set(split['train']) & set(split['dev']))
            args.seed = 44
            self.assertEqual(build_data(args)[-1], split)
            args.eval_protocol = 'test-monitored'
            train, dev, test, _, split = build_data(args)
            self.assertEqual(len(train), 8)
            self.assertIs(dev, test)
            self.assertEqual(split['monitor'], 'test_monitored')

    def test_imagefolder_train_val_and_optional_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for partition in ('train', 'val', 'test'):
                for label in ('a', 'b'):
                    folder = root / partition / label
                    folder.mkdir(parents=True)
                    for i in range(3):
                        Image.fromarray(np.full((12, 12, 3), 10 + i, dtype=np.uint8)).save(folder / f'{i}.png')
            args = parser().parse_args(['--dataset', 'imagefolder', '--data-dir', str(root), '--image-size', '8', '--train-per-class', '2'])
            train, dev, test, count, split = build_data(args)
            self.assertEqual((len(train), len(dev), len(test), count), (4, 6, 6, 2))
            self.assertEqual(train[0][0].shape, (3, 8, 8))
            self.assertEqual(split['train'][0][0].split('/')[0], 'train')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(['--dataset', 'imagefolder', '--data-dir', str(root), '--image-size', '8',
                      '--model-version', 'legacy', '--normalization', 'raw', '--epochs', '1',
                      '--cpu', '--output-dir', str(root / 'run')])
            self.assertTrue((root / 'run/result.json').is_file())
            (root / 'test').rename(root / 'heldout')
            self.assertIsNone(build_data(args)[2])
            (root / 'val/b').rename(root / 'val/c')
            with self.assertRaisesRegex(ValueError, 'identical class'):
                build_data(args)

    def test_legacy_cli_aliases(self):
        args = parser().parse_args(['--no-trace-branch', '--no-edge-enhance', '--directed-flow',
            '--no-cross-guidance', '--use-chebyshev', '--use-spatial-bridge', '--no-aug'])
        self.assertFalse(args.trace or args.edge or args.cross or args.augment)
        self.assertTrue(args.directed and args.chebyshev and args.spatial)

    def test_every_dataset_and_version_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            for dataset in ('mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imagefolder'):
                for version in ('legacy', 'current', 'tiny'):
                    with self.subTest(dataset=dataset, version=version):
                        out = Path(directory) / f'{dataset}_{version}'
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            main(['--dataset', dataset, '--model-version', version, '--smoke', '--cpu',
                                  '--epochs', '1', '--image-size', '8', '--normalization', 'raw',
                                  '--chebyshev', '--spatial', '--cross', '--output-dir', str(out)])
                        self.assertTrue((out / 'last.pth').is_file())


if __name__ == '__main__':
    unittest.main()
