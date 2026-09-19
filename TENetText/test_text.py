import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import train
from text_data import prepare, download
from text_model import TextClassifier, matched_width


class TextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_padding_invariance_and_finite_gradients(self):
        for version in ('tenet', 'cnn', 'mean'):
            for norm in ('raw', 'centered', 'standardized'):
                model = TextClassifier(20, 4, version, 4, 3, norm)
                tokens = torch.tensor([[2, 3, 4], [5, 0, 0], [0, 0, 0]])
                before = model(tokens)
                after = model(torch.nn.functional.pad(tokens, (0, 5)))
                torch.testing.assert_close(before, after)
                before.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertEqual(model.embedding.weight.grad[0].abs().sum().item(), 0)

    def test_embedding_and_budget_matching(self):
        models = []
        for version in ('tenet', 'cnn', 'mean'):
            torch.manual_seed(42)
            width = matched_width(version, 16, 4, 25684)
            model = TextClassifier(100, 4, version, 16, width)
            models.append(model)
            self.assertLessEqual(abs(model.parameter_counts()['backbone_and_head'] - 25684), 35)
        for model in models[1:]:
            torch.testing.assert_close(models[0].embedding.weight, model.embedding.weight)

    def test_download_never_publishes_partial_csv(self):
        class Response(io.BytesIO):
            def __init__(self, body, claimed_size=None):
                super().__init__(body)
                self.headers = {'Content-Length': str(claimed_size or len(body))}
        good = b'"1","title","a"\n"2","title","b"\n"3","title","c"\n"4","title","d"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            with patch('text_data.urllib.request.urlopen', side_effect=lambda *a, **kw: Response(good[:10], len(good))), patch('text_data.time.sleep'), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    download(path, 4)
            self.assertFalse(path.exists())
            with patch('text_data.urllib.request.urlopen', return_value=Response(good)), contextlib.redirect_stderr(io.StringIO()):
                download(path, 4)
            self.assertEqual(path.read_bytes(), good)

    def test_split_vocabulary_and_local_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ('train.csv', 'test.csv'):
                with (root / filename).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['label', 'text'])
                    for c in range(2):
                        for i in range(10):
                            writer.writerow([str(c), f'topic{c} unique{c}_{i}' if filename == 'train.csv' else 'testonlyword'])
            args = train.parser().parse_args(['--dataset', 'csv', '--data-dir', directory])
            a, b, c, vocab, split = prepare(args)
            self.assertEqual((len(a), len(b), len(c)), (18, 2, 20))
            self.assertNotIn('testonlyword', vocab)
            for i in split['dev']:
                self.assertNotIn(f'unique{i // 10}_{i % 10}', vocab)
            self.assertFalse(set(split['train']) & set(split['dev']))
            args.seed = 99
            self.assertEqual(split, prepare(args)[-1])
            args.tokenizer = 'char'
            self.assertIn('t', prepare(args)[3])

    def test_all_models_train_record_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in ('tenet', 'cnn', 'mean'):
                args = ['--smoke', '--cpu', '--epochs', '1', '--max-length', '12', '--embedding-dim', '4',
                        '--hidden-dim', '3', '--model-version', version, '--output-dir', str(root / version),
                        '--records-dir', str(root / 'records')]
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    train.main(args)
                    train.main(args + ['--resume'])
                result = json.loads((root / version / 'result.json').read_text())
                self.assertTrue(result['synthetic'])
                self.assertEqual(result['checkpoint'], 'best.pth')
                self.assertEqual(result['model_version'], version)
            rows = json.loads((root / 'records/training_records.json').read_text())
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['status'] == 'complete' for r in rows))

    def test_interrupted_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['--smoke', '--cpu', '--epochs', '2', '--max-length', '8', '--embedding-dim', '3',
                    '--hidden-dim', '3', '--dropout', '.2', '--records-dir', str(root / 'records')]
            original = train.epoch
            calls = 0
            def interrupted(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt()
                return original(*a, **kw)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                train.main(args + ['--output-dir', str(root / 'full')])
                with patch.object(train, 'epoch', interrupted), self.assertRaises(KeyboardInterrupt):
                    train.main(args + ['--output-dir', str(root / 'resume')])
                train.main(args + ['--output-dir', str(root / 'resume'), '--resume'])
            left, right = [torch.load(root / folder / 'last.pth', weights_only=False) for folder in ('full', 'resume')]
            for key in left['model']:
                torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)
            self.assertEqual(left['best_summary'], right['best_summary'])


if __name__ == '__main__':
    unittest.main()
