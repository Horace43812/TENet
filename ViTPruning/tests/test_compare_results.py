import json
import tempfile
import unittest
from pathlib import Path

from compare_results import load_tail, render_table


def make_run(index):
    return {"run_id": str(index), "configuration": {
        "args": {"pruner": "magnitude", "seed": 42, "epochs": 2, "output_dir": str(index)},
        "data": {"class_names": ["a", "b"], "train_samples": 10, "val_samples": 4},
        "model": {"image_size": 128, "embed_dim": 256}},
        "summary": {"status": "completed", "best_val_accuracy_percent": 75, "best_epoch": 1,
                    "final_val": {"accuracy": 50, "loss": 1.25}, "elapsed_seconds_this_invocation": 3},
        "history": [{"eval": {"accuracy": 75}}, {"eval": {"accuracy": 50}}]}


class CompareResultsTest(unittest.TestCase):
    def test_tail_and_means(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text("\n".join(json.dumps(make_run(i)) for i in range(6)) + "\n\n", encoding="utf-8")
            records = load_tail(path)
            self.assertEqual([r["run_id"] for r in records], ["2", "3", "4", "5"])
            report = render_table(records)
            self.assertIn("62.50", report)
            self.assertIn("repeated pruners", report)
            self.assertNotIn("configurations differ", report)
            records[-1]["configuration"]["args"]["seed"] = 43
            self.assertIn("configurations differ", render_table(records))
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"unfinished":')
            with self.assertRaisesRegex(ValueError, "line"):
                load_tail(path)

    def test_empty_and_short(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tail(path)
            path.write_text(json.dumps(make_run(0)), encoding="utf-8")
            self.assertEqual(len(load_tail(path)), 1)
