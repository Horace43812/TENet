"""Meaningful controls for the integrated research platform; all data are temporary."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from aggregators import Aggregation, ResearchNet, OPERATORS
from experiment import parser, initialize, main
from experiment_data import prepare_manifest, atomic_json, load_manifest, verify_files, fingerprint, make_dataset
from experiment_suite import add_jobs, load_project, presets, project_lock
from experiment_runner import train_job, run_project, freeze_final, evaluate_final, parameter_cost, match_budget, benchmark, check_project, RunIntegrityError
from experiment_report import paired_summary, report_project, project_status
from common import load_checkpoint
from model import TENet, build_model


def create_images(root):
    rng = np.random.default_rng(817)
    for split in ("train", "val"):
        for label in (0, 1):
            folder = root / split / str(label)
            folder.mkdir(parents=True)
            for index in range(8 if split == "train" else 3):
                pixels = rng.integers(0, 80, (12, 12, 3), dtype=np.uint8)
                pixels[:, :, label] += 120
                Image.fromarray(pixels).save(folder / f"{index}.png")


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_operators_have_finite_gradients_and_no_unused_parameters(self):
        for op in OPERATORS:
            with self.subTest(op=op):
                layer = Aggregation(3, op, residual=True)
                x = torch.randn(2, 3, 8, 8, requires_grad=True)
                result = layer(x)
                self.assertEqual(result.shape, (2, 3, 4, 4))
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in layer.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertFalse(hasattr(layer, "content_proj"))

    def test_linear_vertex_matches_stride_two_convolution(self):
        layer = Aggregation(3, "V").eval()
        conv = nn.Conv2d(3, 3, 2, stride=2)
        with torch.no_grad():
            conv.weight.copy_(layer.projection.weight.reshape_as(conv.weight))
            conv.bias.copy_(layer.projection.bias)
        x = torch.randn(2, 3, 8, 12)
        torch.testing.assert_close(layer(x), torch.relu(layer.norm(conv(x))))

    def test_graph_formula_and_normalization_reconstruction(self):
        layer = Aggregation(3, "G").eval()
        x = torch.randn(2, 3, 4, 6)
        baseline = layer(x)
        values = {}
        layer.observer = lambda name, value: values.update({name: value})
        layer.capture_normalization = True
        torch.testing.assert_close(layer(x), baseline)
        nodes = values["nodes"][0, :, 0, 0].reshape(3, 10)
        torch.testing.assert_close(nodes[:, :4], x[0, :, :2, :2].reshape(3, 4))
        graph = nodes.unsqueeze(-1) * layer.L * nodes.unsqueeze(-2)
        torch.testing.assert_close(values["graph_raw"][0, :, 0, 0].reshape(3, 10, 10), graph)
        torch.testing.assert_close(values["graph_restored"], values["graph_raw"], rtol=1e-5, atol=1e-5)
        for mode in ("centered", "raw"):
            layer.normalization = mode
            layer.capture_normalization = False
            layer(x)
            raw = values["graph_raw"].reshape(2, 3, 100, 2, 3)
            expected = raw if mode == "raw" else raw - raw.mean(2, keepdim=True)
            torch.testing.assert_close(values["graph_normalized"].reshape_as(raw), expected)

    def test_mismatch_preserves_degree_changes_semantic_assignment(self):
        original = Aggregation(3, "G").L
        matrices = []
        for seed in (1, 2, 3):
            changed = Aggregation(3, "mismatch", mismatch_seed=seed).L
            torch.testing.assert_close(changed.diag(), original.diag())
            torch.testing.assert_close(changed, changed.T)
            self.assertEqual(int((changed < 0).sum()), 24)
            self.assertFalse(torch.equal(changed, original))
            matrices.append(changed)
        self.assertTrue(all(not torch.equal(matrices[a], matrices[b]) for a in range(3) for b in range(a)))

    def test_factory_legacy_and_schedule(self):
        old = TENet(2, image_size=8, layers=1, head_grid=1)
        restored = build_model(dict(num_classes=2, image_size=8, layers=1, head_grid=1))
        restored.load_state_dict(old.state_dict())
        x = torch.randn(2, 3, 8, 8)
        torch.testing.assert_close(restored.eval()(x), old.eval()(x))
        for schedule in ("DDDPPP", "DPDPDP"):
            model = ResearchNet(2, image_size=8, channels=3, head_grid=1, schedule=schedule)
            self.assertEqual(model(x).shape, (2, 2))


class ProjectTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        create_images(self.data)

    def tearDown(self):
        self.temporary.cleanup()

    def init(self, epochs=1):
        project = self.root / "project"
        args = parser().parse_args(["init", "--project", str(project), "--data-dir", str(self.data),
            "--image-size", "8", "--head-grid", "1", "--head-hidden", "8", "--width", "3",
            "--epochs", str(epochs), "--batch-size", "2", "--batch", "p1", "--models", "G,Q"])
        with project_lock(project):
            initialize(args)
        return project

    def test_manifest_is_fixed_verified_and_excludes_final(self):
        manifest = prepare_manifest(self.data)
        self.assertEqual(manifest, prepare_manifest(self.data))
        paths = [{r["path"] for r in manifest["splits"][key]} for key in ("train", "dev", "final")]
        self.assertFalse(paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
        final = self.data / manifest["splits"]["final"][0]["path"]
        final.write_bytes(b"changed final")
        verify_files(manifest, self.data)  # training checks must not access final
        with self.assertRaises(ValueError):
            verify_files(manifest, self.data, ("final",))
        path = self.root / "manifest.json"
        atomic_json(path, manifest)
        load_manifest(path)
        manifest["class_names"].reverse()
        atomic_json(path, manifest)
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_duplicates_are_retained_with_original_labels(self):
        source = self.data / "train/0/0.png"
        (self.data / "train/1/duplicate.png").write_bytes(source.read_bytes())
        (self.data / "val/0/duplicate.png").write_bytes(source.read_bytes())
        manifest = prepare_manifest(self.data)
        rows = {r["path"]: r for split in manifest["splits"].values() for r in split}
        for path, label in (("train/0/0.png", 0), ("train/1/duplicate.png", 1), ("val/0/duplicate.png", 0)):
            self.assertEqual(rows[path]["label"], label)
            self.assertEqual(rows[path]["sha256"], rows["train/0/0.png"]["sha256"])
        verify_files(manifest, self.data, ("train", "dev", "final"))

    def test_source_group_leak_is_rejected(self):
        mapping = self.root / "groups.json"
        atomic_json(mapping, {"train/0/0.png": "same_subject", "val/0/0.png": "same_subject"})
        with self.assertRaisesRegex(ValueError, "Source groups"):
            prepare_manifest(self.data, groups_file=mapping)

    def test_suite_budget_append_and_plan_tampering(self):
        project = self.init()
        _, settings, manifest, plan = load_project(project)
        self.assertEqual(len(presets(settings, "core")), 16)
        self.assertEqual(len(presets(settings, "positions")), 8)
        new = add_jobs(project, "confirm", (43, 44), ["G", "Q"])
        self.assertEqual(len(new), 4)
        self.assertEqual(add_jobs(project, "confirm", (43, 44), ["G", "Q"]), [])
        with self.assertRaises(RuntimeError):
            with project_lock(project), project_lock(project):
                pass
        _, _, _, plan = load_project(project)
        plan["jobs"][0]["seed"] = 999
        atomic_json(project / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "edited"):
            load_project(project)

    def test_resume_restores_training_rng_and_atomic_completion(self):
        project = self.init(epochs=2)
        _, settings, frozen, plan = load_project(project)
        baseline = plan["jobs"][0]
        train_job(project, settings, frozen, baseline, torch.device("cpu"), 0)
        repeated = {**baseline, "run_id": "resume_check"}
        import experiment_runner
        real_pass = experiment_runner.pass_model
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt()
            return real_pass(*args, **kwargs)
        with patch("experiment_runner.pass_model", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                train_job(project, settings, frozen, repeated, torch.device("cpu"), 0)
        checkpoint = load_checkpoint(project / "runs/resume_check/last.pth", "cpu")
        self.assertEqual(checkpoint["epoch"], 1)
        train_job(project, settings, frozen, repeated, torch.device("cpu"), 0)
        expected = load_checkpoint(project / "runs" / baseline["run_id"] / "last.pth", "cpu")
        actual = load_checkpoint(project / "runs/resume_check/last.pth", "cpu")
        for key in expected["model"]:
            torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)

    def test_final_requires_frozen_complete_runs_and_caches(self):
        project = self.init()
        with self.assertRaises(FileNotFoundError):
            evaluate_final(project, cpu=True)
        run_project(project, cpu=True)
        run_project(project, cpu=True)
        freeze_final(project, ["G", "Q"], [42])
        with self.assertRaises(ValueError):
            add_jobs(project, "confirm", (43,), ["G"])
        with self.assertRaises(ValueError):
            run_project(project, cpu=True)
        evaluate_final(project, cpu=True)
        result = project / "runs/G_s42/evaluation_final.json"
        modified = result.stat().st_mtime_ns
        evaluate_final(project, cpu=True)
        self.assertEqual(modified, result.stat().st_mtime_ns)
        self.assertTrue(report_project(project).exists())

    def test_cost_and_paired_statistics_do_not_claim_single_seed_support(self):
        model = ResearchNet(2, image_size=8, channels=3, head_grid=1, head_hidden=8)
        costs = parameter_cost(model, 8, torch.device("cpu"))
        self.assertFalse(costs["complete_flops"])
        self.assertGreater(costs["conv_linear_macs"], 0)
        self.assertEqual(costs["parameters"], costs["head_parameters"] + costs["backbone_parameters"])
        self.assertIsNone(paired_summary([1.])['ci_low'])
        self.assertEqual(paired_summary([1., 1., 1.])["conclusion"], "supports improvement")

    def test_budget_selection_never_reads_accuracy_and_schedules_only_matches(self):
        project = self.init()
        match_budget(project, cpu=True)
        result = json.loads((project / "budget_selection_parameters.json").read_text(encoding="utf-8"))
        _, _, _, plan = load_project(project)
        names = {j["model_id"] for j in plan["jobs"]}
        for row in result["comparisons"]:
            self.assertIn(row["target"], names)
            if row["matched"]:
                self.assertLessEqual(row["relative_gap"], .05)
                self.assertIn(row["candidate"], names)
        self.assertFalse(any((project / "runs").glob("*/evaluation_dev.json")))

    def test_benchmark_records_trials_and_dashboard(self):
        project = self.init()
        run_project(project, cpu=True, models=["Q"])
        benchmark(project, models=["Q"], cpu=True, iterations=2, warmup=1, repeats=2)
        measurement = next((project / "runs/Q_s42/benchmarks").glob("*.json"))
        data = json.loads(measurement.read_text(encoding="utf-8"))
        self.assertEqual(len(data["measurements"]), 4)
        self.assertTrue(all(r["median_ms"] > 0 for r in data["measurements"]))
        self.assertTrue(all(r["peak_allocated_bytes"] is None for r in data["measurements"]))
        report_project(project)
        self.assertTrue((project / "report/benchmark.csv").exists())

    def test_status_is_readable_during_training_lock_and_intermediate_excluded(self):
        project = self.init()
        _, _, _, plan = load_project(project)
        job = plan["jobs"][0]
        run = project / "runs" / job["run_id"]
        atomic_json(run / "state.json", {"status": "running", "fingerprint": job["fingerprint"]})
        atomic_json(run / "metrics.json", [{"epoch": 1, "val": {"accuracy": 99.}}])
        with project_lock(project):
            main(["status", "--project", str(project)])
        self.assertEqual(project_status(project)[0]["dev_accuracy"], 99.)
        report_project(project)
        report = json.loads((project / "report/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["summary"], [])

    def test_invalid_initial_model_leaves_project_retryable(self):
        project = self.root / "invalid"
        args = parser().parse_args(["init", "--project", str(project), "--models", "misspelled"])
        with project_lock(project):
            with self.assertRaisesRegex(ValueError, "Unknown models"):
                initialize(args)
        self.assertEqual([p.name for p in project.iterdir()], [".lock"])

    def test_integrity_failure_preserves_completion_metadata_and_blocks_analysis(self):
        project = self.init()
        run_project(project, cpu=True, models=["Q"])
        run = project / "runs/Q_s42"
        original = (run / "state.json").read_bytes()
        original_checkpoint = (run / "last.pth").read_bytes()
        (run / "last.pth").write_bytes(b"corrupt")
        with self.assertRaises(RunIntegrityError):
            run_project(project, cpu=True, models=["Q"])
        self.assertEqual((run / "state.json").read_bytes(), original)
        self.assertEqual(next(r for r in project_status(project) if r["run_id"] == "Q_s42")["status"], "integrity_error")
        (run / "last.pth").write_bytes(original_checkpoint)
        run_project(project, cpu=True, models=["Q"])
        self.assertEqual(next(r for r in project_status(project) if r["run_id"] == "Q_s42")["status"], "complete")

    def test_start_reuses_project_and_check_never_loads_final(self):
        project = self.init()
        _, _, manifest, _ = load_project(project)
        final_path = self.data / manifest["splits"]["final"][0]["path"]
        final_path.write_bytes(b"unavailable final")
        self.assertFalse(check_project(project)["final_loaded"])
        main(["start", "--project", str(project), "--models", "Q", "--cpu", "--threads", "1"])
        checkpoint = project / "runs/Q_s42/last.pth"
        modified = checkpoint.stat().st_mtime_ns
        main(["start", "--project", str(project), "--models", "Q", "--cpu", "--threads", "1"])
        self.assertEqual(checkpoint.stat().st_mtime_ns, modified)


if __name__ == "__main__":
    unittest.main()
