import json
import tempfile
import unittest
from pathlib import Path

from results import append_result


class ResultsTest(unittest.TestCase):
    def test_append_preserves_previous_run_and_records_best(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.jsonl"
            history = [{"epoch": epoch, "lr": 0.001, "train": {"loss": 1, "accuracy": 50},
                        "eval": {"loss": 1, "accuracy": accuracy}}
                       for epoch, accuracy in [(1, 70), (2, 60)]]
            options = dict(args={"epochs": 2}, model_config={}, data_config={"class_names": ["a", "b"]},
                           history=history, started_at="2026-01-01T00:00:00+00:00", elapsed_seconds=3,
                           device="cpu", parameter_count=100, start_epoch=2)
            append_result(path, **options)
            original = path.read_bytes()
            append_result(path, **options)
            self.assertTrue(path.read_bytes().startswith(original))
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]["run_id"], records[1]["run_id"])
            self.assertEqual(records[1]["summary"]["best_epoch"], 1)
            self.assertEqual(records[1]["summary"]["epochs_trained_this_invocation"], 1)
            self.assertEqual(records[1]["history"], history)
