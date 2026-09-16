import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from character_data import Kuzushiji49, UprightEMNIST
from data import build_data, class_count_hint
from records import RunRecord
from runtime import epoch
import train
import train_v1


class Letters:
    def __init__(self, root, train=True, download=False, transform=None, split=None):
        assert split == 'letters'
        self.targets = torch.tensor([label for label in range(1, 27) for _ in range(4 if train else 1)])
        self.classes = ['N/A'] + [chr(97 + i) for i in range(26)]
        self.transform = transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        pixels = np.zeros((28, 28), dtype=np.uint8)
        pixels[2, 7] = 255
        return self.transform(Image.fromarray(pixels)), int(self.targets[index])


class Progress:
    instances = []
    def __init__(self, total, initial=0, **kwargs):
        self.total, self.n, self.initial = total, initial, initial
        self.instances.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, n): self.n += n
    def set_description(self, *args): pass
    def set_postfix(self, **kwargs): pass


class CharacterRecordTests(unittest.TestCase):
    def test_emnist_letters_labels_and_orientation(self):
        args = train.parser().parse_args(['--dataset', 'emnist', '--emnist-split', 'letters', '--dev-per-class', '1'])
        with patch.dict('data.SPECS', {'emnist': (Letters, 1, 28, (.5,), (.5,), 500)}):
            training, dev, test, count, manifest = build_data(args)
        self.assertEqual((len(training), len(dev), len(test), count), (78, 26, 26, 26))
        self.assertEqual(class_count_hint(args), 26)
        self.assertEqual(set(training.dataset.targets.tolist()), set(range(26)))
        image, label = test[0]
        self.assertEqual(label, 0)
        self.assertEqual(image[0, 7, 2].item(), 1.)
        self.assertFalse(set(manifest['train']) & set(manifest['dev']))

    def test_k49_official_npz_and_sparse_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'K49'
            folder.mkdir()
            for part in ('train', 'test'):
                labels = np.array([label for label in range(49) for _ in range(4 if part == 'train' else 1)], dtype=np.uint8)
                np.savez(folder / f'k49-{part}-imgs.npz', np.zeros((len(labels), 28, 28), dtype=np.uint8))
                np.savez(folder / f'k49-{part}-labels.npz', labels)
            args = train.parser().parse_args(['--dataset', 'k49', '--data-dir', directory])
            training, dev, test, count, manifest = build_data(args)
            self.assertEqual((len(training), len(dev), len(test), count), (147, 49, 49, 49))
            self.assertFalse(set(manifest['train']) & set(manifest['dev']))
            self.assertEqual(test[48][1], 48)

    def test_balanced_accuracy_is_not_top1(self):
        labels = torch.tensor([0] * 9 + [1])
        logits = torch.tensor([[1., 0.]] * 10)
        result = epoch(torch.nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=4), torch.device('cpu'))
        self.assertEqual(result['accuracy_pct'], 90.)
        self.assertEqual(result['balanced_accuracy_pct'], 50.)

    def test_records_summary_progress_interrupt_resume(self):
        self.assertEqual(train.parser().parse_args([]).epochs, 50)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['--smoke', '--cpu', '--epochs', '2', '--image-size', '8', '--output-dir', str(root / 'run'), '--records-dir', str(root / 'records')]
            original = train.epoch
            calls = 0
            def fail(*a, **k):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt()
                return original(*a, **k)
            Progress.instances = []
            with contextlib.redirect_stdout(io.StringIO()), patch('runtime.tqdm', Progress):
                with patch.object(train, 'epoch', fail), self.assertRaises(KeyboardInterrupt):
                    train.main(args)
                rows = json.loads((root / 'records/training_records.json').read_text(encoding='utf-8'))
                self.assertEqual(rows[0]['status'], 'interrupted')
                self.assertEqual(rows[0]['completed_epochs'], 1)
                train.main(args + ['--resume'])
            rows = json.loads((root / 'records/training_records.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'complete')
            self.assertIn('best_summary', rows[0])
            self.assertNotIn('history', rows[0])
            self.assertFalse((root / 'run/metrics.json').exists())
            self.assertEqual([(p.total, p.initial, p.n) for p in Progress.instances], [(5, 0, 2), (5, 2, 5)])

    def test_records_failure_and_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            def write(index):
                with RunRecord(folder, folder / f'run{index}', {'seed': index}) as record:
                    record.update(status='complete')
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(write, range(8)))
            with self.assertRaisesRegex(RuntimeError, 'broken'):
                with RunRecord(folder, folder / 'failed', {}):
                    raise RuntimeError('broken')
            rows = json.loads((folder / 'training_records.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows[-1]['status'], 'failed')

    def test_v1_uses_same_registry_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['train_v1.py', '--smoke', '--cpu', '--epochs', '1', '--output-dir', str(root / 'run'), '--records-dir', str(root / 'records')]
            with patch('sys.argv', argv), patch('runtime.tqdm', Progress), contextlib.redirect_stdout(io.StringIO()):
                train_v1.main()
            rows = json.loads((root / 'records/training_records.json').read_text(encoding='utf-8'))
            self.assertEqual(rows[0]['status'], 'complete')
            self.assertEqual(rows[0]['parameters'], 2295)
            self.assertFalse((root / 'run/metrics.json').exists())

    def test_previous_release_history_migrates_without_retraining_completed_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['--smoke', '--cpu', '--epochs', '2', '--image-size', '8', '--output-dir', str(root / 'run'), '--records-dir', str(root / 'records')]
            original = train.epoch
            calls = 0
            def fail(*a, **k):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt()
                return original(*a, **k)
            with patch('runtime.tqdm', Progress), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(train, 'epoch', fail), self.assertRaises(KeyboardInterrupt):
                    train.main(args)
                path = root / 'run/last.pth'
                state = torch.load(path, weights_only=False)
                old = state.pop('last_summary')
                state.pop('best_summary')
                state['history'] = [{'epoch': old['epoch'], 'lr': old['lr'], 'train': old['train'], 'dev': old['validation']}]
                state['config'].pop('emnist_split')
                state['config'].pop('selection_metric')
                state['config']['source_sha256'] = sorted(train.PREVIOUS_SOURCES)[0]
                torch.save(state, path)
                train.main(args + ['--resume'])
            finished = torch.load(path, weights_only=False)
            self.assertEqual(finished['epoch'], 2)
            self.assertNotIn('history', finished)
            self.assertIn('best_summary', finished)
            changed = dict(finished['config'], lr=.2)
            self.assertFalse(train.compatible_resume(state['config'], changed))

    def test_failed_download_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['--dataset', 'emnist', '--output-dir', str(root / 'run'), '--records-dir', str(root / 'records')]
            with patch('train.build_data', side_effect=OSError('download unavailable')), self.assertRaises(OSError):
                train.main(args)
            rows = json.loads((root / 'records/training_records.json').read_text(encoding='utf-8'))
            self.assertEqual(rows[0]['status'], 'failed')
            self.assertIn('download unavailable', rows[0]['error'])


if __name__ == '__main__':
    unittest.main()
