import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from cnn import CNN
from train import main, compatible_resume, CHARACTER_RELEASE_SOURCES


class CNNTests(unittest.TestCase):
    def test_budget_and_gradients(self):
        torch.set_num_threads(1)
        for classes, count in ((10, 2398), (47, 7874)):
            model = CNN(num_classes=classes)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            model(torch.randn(2, 1, 28, 28)).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model = CNN(in_channels=3, image_size=31, num_classes=100, channels=[8, 16, 24], head='mlp')
        self.assertEqual(model(torch.randn(2, 3, 31, 31)).shape, (2, 100))

    def test_all_datasets_train_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            for dataset in ('mnist', 'fashionmnist', 'emnist', 'kmnist', 'k49', 'cifar10', 'cifar100', 'imagefolder'):
                with self.subTest(dataset=dataset):
                    out = Path(directory) / dataset
                    args = ['--dataset', dataset, '--model-version', 'cnn', '--smoke', '--cpu',
                            '--epochs', '1', '--image-size', '8', '--output-dir', str(out),
                            '--records-dir', str(Path(directory) / 'records')]
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        main(args)
                        before = (out / 'result.json').read_bytes()
                        main(args + ['--resume'])
                    self.assertEqual(before, (out / 'result.json').read_bytes())
                    result = json.loads(before)
                    self.assertEqual(result['model_version'], 'cnn')
                    self.assertEqual(result['normalization'], 'not_applicable')
                    self.assertEqual(result['selection_metric'], 'balanced_accuracy_pct' if dataset == 'k49' else 'accuracy_pct')

    def test_graph_controls_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for flag in ('--no-directed', '--normalization=raw', '--window-layout=corrected'):
                with self.assertRaises(SystemExit):
                    main(['--model-version', 'cnn', '--describe', flag])

    def test_previous_release_resume_still_checks_config(self):
        previous = dict(source_sha256=next(iter(CHARACTER_RELEASE_SOURCES)), model_version='current', lr=.001)
        current = dict(previous, source_sha256='new')
        self.assertTrue(compatible_resume(previous, current))
        self.assertFalse(compatible_resume(previous, dict(current, lr=.01)))


if __name__ == '__main__':
    unittest.main()
