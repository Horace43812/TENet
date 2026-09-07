"""Resumable training, selected-checkpoint analysis, and separately frozen final evaluation."""
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import seed_all, load_checkpoint
from datasets import _seed_worker
from experiment_data import atomic_json, read_json, fingerprint, digest_file, make_dataset, verify_files
from experiment_suite import load_project, source_fingerprint
from model import build_model


class RunIntegrityError(ValueError):
    """A validation failure must not overwrite trusted completion metadata."""


def validate_completed(run, job):
    path = run / "state.json"
    if not path.exists():
        return False
    state = read_json(path)
    if state.get("fingerprint") != job["fingerprint"]:
        raise RunIntegrityError(f"Run directory contains another configuration: {job['run_id']}")
    if state["status"] != "complete":
        return False
    checkpoint = run / "last.pth"
    if not checkpoint.exists() or digest_file(checkpoint) != state.get("checkpoint_sha256"):
        raise RunIntegrityError(f"Completed checkpoint is missing or changed: {job['run_id']}")
    if (run / "integrity.json").exists():
        atomic_json(run / "integrity.json", {"status": "passed"})
    return True


def check_project(project):
    project, settings, frozen, plan = load_project(project)
    if source_fingerprint() != settings["source_fingerprint"]:
        raise ValueError("Code differs from the project snapshot; use the matching source version or a new project")
    verify_files(frozen, settings["data_dir"])
    complete = 0
    for job in plan["jobs"]:
        expected = fingerprint({"model": job["model_config"], "seed": job["seed"], "project": settings["fingerprint"]})
        if job["fingerprint"] != expected:
            raise RunIntegrityError(f"Job fingerprint mismatch: {job['run_id']}")
        build_model(job["model_config"])
        complete += validate_completed(project / "runs" / job["run_id"], job)
    result = dict(runs=len(plan["jobs"]), complete=complete,
        train_samples=len(frozen["splits"]["train"]), dev_samples=len(frozen["splits"]["dev"]),
        free_disk_gib=round(shutil.disk_usage(project).free / 1024 ** 3, 2), final_loaded=False)
    print("CHECK PASS:", result, flush=True)
    return result


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def environment(device):
    import torchvision
    return dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
        platform=platform.platform(), device=str(device), device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda" else platform.processor(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(), dtype="float32", mode="eager")


def loader(dataset, batch, workers, seed, shuffle=False):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
        worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(), drop_last=shuffle and len(dataset) % batch == 1)


def pass_model(model, batches, device, optimizer=None, smoothing=.1, predictions=False, progress_desc=None):
    model.train(optimizer is not None)
    correct = count = 0
    total_loss = nll = 0.
    labels, guesses = [], []
    with torch.set_grad_enabled(optimizer is not None), tqdm(
        batches, desc=progress_desc, disable=progress_desc is None,
        file=sys.stdout, dynamic_ncols=True, mininterval=1.0, unit="batch", ascii=True
    ) as progress:
        for images, target in progress:
            images, target = images.to(device), target.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, target, label_smoothing=smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
            guessed = logits.argmax(1)
            correct += int((guessed == target).sum())
            count += len(target)
            total_loss += float(loss.detach()) * len(target)
            nll += float(nn.functional.cross_entropy(logits.detach(), target, reduction="sum"))
            progress.set_postfix(loss=f"{total_loss / count:.4f}", acc=f"{100 * correct / count:.2f}%", refresh=False)
            if predictions:
                labels.extend(target.cpu().tolist())
                guesses.extend(guessed.cpu().tolist())
    if not count:
        raise ValueError("Empty evaluation dataset")
    result = dict(loss=total_loss / count, nll=nll / count, accuracy=100 * correct / count, samples=count)
    if predictions:
        classes = logits.shape[1]
        confusion = torch.zeros(classes, classes, dtype=torch.int64)
        for label, guess in zip(labels, guesses):
            confusion[label, guess] += 1
        result.update(labels=labels, predictions=guesses, confusion=confusion.tolist(),
            macro_recall=float((confusion.diag() / confusion.sum(1).clamp_min(1)).mean() * 100))
    return result


def parameter_cost(model, image_size, device):
    """Count executed dense Conv/Linear MACs; explicitly excludes graph arithmetic."""
    macs = 0
    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, image_size, image_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.classifier.parameters())
    return dict(parameters=total, head_parameters=head, backbone_parameters=total - head,
        conv_linear_macs=int(macs), conv_linear_flops=int(2 * macs), complete_flops=False,
        excluded="Elementwise graph construction, normalization, activations, pooling and data movement")


def train_job(project, settings, frozen, job, device, workers):
    output = project / "runs" / job["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    if validate_completed(output, job):
        print(f"SKIP {job['run_id']} (complete)", flush=True)
        return
    atomic_json(state_path, dict(status="running", fingerprint=job["fingerprint"], pid=os.getpid()))
    from experiment_report import report_project
    report_project(project)
    seed_all(job["seed"])
    train_set = make_dataset(frozen, settings["data_dir"], "train", settings["image_size"], True)
    dev_set = make_dataset(frozen, settings["data_dir"], "dev", settings["image_size"])
    batches = loader(train_set, settings["batch_size"], workers, job["seed"], True)
    dev_batches = loader(dev_set, settings["batch_size"], workers, job["seed"])
    model = build_model(job["model_config"]).to(device)
    cost = parameter_cost(model, settings["image_size"], device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{job['run_id']} | Device: {device} | Parameters: {cost['parameters']:,} | "
          f"Trainable: {trainable:,} | Backbone: {cost['backbone_parameters']:,} | Head: {cost['head_parameters']:,}", flush=True)
    atomic_json(output / "cost.json", cost)
    runtime = environment(device)
    atomic_json(output / "environment.json", runtime)
    atomic_json(output / "sessions" / f"{time.time_ns()}.json", runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
    args = dict(dataset="imagefolder", frozen_manifest=str(project / "manifest.json"), data_dir=settings["data_dir"],
        image_size=settings["image_size"], batch_size=settings["batch_size"], workers=workers, seed=job["seed"], no_aug=False)
    samples = {"class_names": frozen["class_names"], "train": [{"path": p, "label": l} for p, l in train_set.samples],
               "val": [{"path": p, "label": l} for p, l in dev_set.samples]}
    history, best, start = [], -1., 1
    checkpoint_path = output / "last.pth"
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, "cpu")
        if checkpoint.get("job_fingerprint") != job["fingerprint"]:
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best, start = checkpoint["history"], checkpoint["best_accuracy"], checkpoint["epoch"] + 1
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        batches.generator.set_state(rng["loader"])
        if device.type == "cuda" and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"RESUME {job['run_id']} epoch {start}", flush=True)
    atomic_json(output / "config.json", dict(job=job, project=settings, args=args, model_config=job["model_config"]))
    atomic_json(output / "manifest.json", samples)
    for epoch in range(start, settings["epochs"] + 1):
        began = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        description = f"{job['run_id']} Epoch {epoch}/{settings['epochs']}"
        train = pass_model(model, batches, device, optimizer, progress_desc=f"{description} [Train]")
        dev = pass_model(model, dev_batches, device, progress_desc=f"{description} [Dev]")
        elapsed = time.perf_counter() - began
        history.append(dict(epoch=epoch, train=train, val=dev, lr=optimizer.param_groups[0]["lr"], seconds=elapsed))
        if device.type == "cuda":
            history[-1]["train_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        scheduler.step()
        improved = dev["accuracy"] > best
        best = max(best, dev["accuracy"])
        checkpoint = dict(format_version=2, epoch=epoch, model_config=job["model_config"], args=args,
            manifest=samples, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            history=history, best_accuracy=best, job_fingerprint=job["fingerprint"], source_fingerprint=settings["source_fingerprint"],
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     loader=batches.generator.get_state(), cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        # Checkpoint commits the epoch; JSON can be reconstructed after an interruption.
        if improved:
            atomic_torch(output / "best.pth", checkpoint)
        atomic_torch(checkpoint_path, checkpoint)
        atomic_json(output / "metrics.json", history)
        atomic_json(state_path, dict(status="running", fingerprint=job["fingerprint"], pid=os.getpid(), epoch=epoch))
        report_project(project)
        print(f"{description} | train loss={train['loss']:.4f} acc={train['accuracy']:.2f}% | "
              f"dev loss={dev['loss']:.4f} acc={dev['accuracy']:.2f}% | {elapsed:.1f}s", flush=True)
    # Handles a crash after the final epoch checkpoint but before completion metadata.
    atomic_json(output / "metrics.json", history)
    result = pass_model(model, dev_batches, device, smoothing=0, predictions=True)
    result.update(checkpoint="last", split="dev", final_status=frozen["final_status"], run_id=job["run_id"])
    atomic_json(output / "evaluation_dev.json", result)
    atomic_json(state_path, dict(status="complete", fingerprint=job["fingerprint"], epoch=settings["epochs"],
                                 checkpoint_sha256=digest_file(checkpoint_path)))


def selected_jobs(plan, models=None, seeds=None):
    if models and set(models) - {j["model_id"] for j in plan["jobs"]}:
        raise ValueError("Requested model not present in project plan")
    jobs = [j for j in plan["jobs"] if (not models or j["model_id"] in models) and (not seeds or j["seed"] in seeds)]
    if not jobs:
        raise ValueError("No matching runs")
    return jobs


def run_project(project, cpu=False, workers=0, models=None, seeds=None, keep_going=False):
    project, settings, frozen, plan = load_project(project)
    if (project / "final_registration.json").exists():
        raise ValueError("Training is locked after final registration")
    if source_fingerprint() != settings["source_fingerprint"]:
        raise ValueError("Platform code changed since init; create a new project to keep versions explicit")
    verify_files(frozen, settings["data_dir"])
    jobs = selected_jobs(plan, models, seeds)
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    print(f"{len(jobs)} selected runs | {settings['epochs']} epochs/run | device={device} | final set excluded", flush=True)
    failures = []
    for job in jobs:
        try:
            train_job(project, settings, frozen, job, device, workers)
        except (Exception, KeyboardInterrupt) as error:
            output = project / "runs" / job["run_id"]
            output.mkdir(parents=True, exist_ok=True)
            failure = dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                fingerprint=job["fingerprint"], error=str(error), error_type=type(error).__name__)
            atomic_json(output / "failures" / f"{time.time_ns()}.json", failure)
            if not isinstance(error, RunIntegrityError):
                atomic_json(output / "state.json", failure)
            else:
                atomic_json(output / "integrity.json", {"status": "failed", "error": str(error)})
            failures.append(job["run_id"])
            if isinstance(error, KeyboardInterrupt) or not keep_going:
                raise
        finally:
            from experiment_report import report_project
            report_project(project)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"Failed runs: {failures}; inspect state.json and rerun to resume")


def analyze_project(project, models=None, seeds=None, cpu=False, probe_epochs=30, probe_dim=256):
    project, settings, frozen, plan = load_project(project)
    if source_fingerprint() != settings["source_fingerprint"]:
        raise ValueError("Analysis code changed since project initialization; use an explicit versioned analysis workflow")
    for job in selected_jobs(plan, models, seeds):
        run = project / "runs" / job["run_id"]
        if not validate_completed(run, job):
            print(f"Analysis skipped (not complete): {job['run_id']}")
            continue
        config = job["model_config"]
        model = build_model(config)
        for index, layer in enumerate(model.layers):
            is_graph = getattr(layer, "is_graph", True)
            norm = getattr(layer, "normalization", "standardized")
            modes = ["flow"] + (["normalization"] if is_graph and norm == "standardized" else [])
            for mode in modes:
                output = run / "analysis" / f"layer{index}_{mode}_e{probe_epochs}_d{probe_dim}"
                if (output / "report.json").exists():
                    continue
                # Preserve incomplete artifacts instead of overwriting.
                if output.exists():
                    output.rename(output.with_name(output.name + f"_interrupted_{time.time_ns()}"))
                command = [sys.executable, str(Path(__file__).with_name("analyze.py")), "--checkpoint", str(run / "last.pth"),
                    "--layer", str(index), "--grid", str(min(4, settings["image_size"] // 2 ** (index + 1))),
                    "--probe-epochs", str(probe_epochs), "--probe-dim", str(probe_dim), "--output-dir", str(output)]
                if mode == "normalization":
                    command.append("--normalization-study")
                if cpu:
                    command.append("--cpu")
                child_env = {**os.environ, "OMP_NUM_THREADS": str(torch.get_num_threads()),
                             "MKL_NUM_THREADS": str(torch.get_num_threads())}
                subprocess.run(command, check=True, env=child_env)


def freeze_final(project, models, seeds):
    project, settings, frozen, plan = load_project(project)
    jobs = selected_jobs(plan, models, seeds)
    if {(j["model_id"], j["seed"]) for j in jobs} != {(m, s) for m in models for s in seeds}:
        raise ValueError("All requested models must have every requested seed")
    records = []
    for job in jobs:
        run = project / "runs" / job["run_id"]
        if not validate_completed(run, job):
            raise ValueError(f"Incomplete run: {job['run_id']}")
        sha = digest_file(run / "last.pth")
        records.append(dict(run_id=job["run_id"], model_id=job["model_id"], seed=job["seed"], sha256=sha))
    registration = dict(project_fingerprint=settings["fingerprint"], records=records,
                        final_status=frozen["final_status"], checkpoint="last")
    registration["fingerprint"] = fingerprint(registration)
    path = project / "final_registration.json"
    if path.exists() and read_json(path) != registration:
        raise ValueError("Final registration already frozen with other comparisons")
    atomic_json(path, registration)
    print(f"Frozen {len(records)} checkpoints; final status: {frozen['final_status']}. Next: final")


def evaluate_final(project, cpu=False):
    project, settings, frozen, plan = load_project(project)
    if source_fingerprint() != settings["source_fingerprint"]:
        raise ValueError("Evaluation code changed since project initialization")
    registration = read_json(project / "final_registration.json")
    if registration["fingerprint"] != fingerprint({k: v for k, v in registration.items() if k != "fingerprint"}):
        raise ValueError("Final registration was edited")
    if registration["project_fingerprint"] != settings["fingerprint"]:
        raise ValueError("Registration/project mismatch")
    verify_files(frozen, settings["data_dir"], ("final",))
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    dataset = make_dataset(frozen, settings["data_dir"], "final", settings["image_size"])
    for record in registration["records"]:
        run = project / "runs" / record["run_id"]
        path = run / "last.pth"
        if digest_file(path) != record["sha256"]:
            raise ValueError("Frozen checkpoint changed")
        output = run / "evaluation_final.json"
        if output.exists():
            if read_json(output).get("checkpoint_sha256") != record["sha256"]:
                raise ValueError("Cached final result refers to another checkpoint")
            continue
        checkpoint = load_checkpoint(path, "cpu")
        model = build_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        result = pass_model(model, loader(dataset, settings["batch_size"], 0, record["seed"]), device,
                            smoothing=0, predictions=True)
        result.update(split="final", checkpoint_sha256=record["sha256"], final_status=frozen["final_status"])
        atomic_json(output, result)
        print(f"Final {record['run_id']}: {result['accuracy']:.2f}% ({frozen['final_status']})")


def benchmark(project, models=None, seeds=None, cpu=False, iterations=200, warmup=50, repeats=5):
    project, settings, frozen, plan = load_project(project)
    if source_fingerprint() != settings["source_fingerprint"]:
        raise ValueError("Benchmark code changed since project initialization")
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    for job in selected_jobs(plan, models, seeds):
        run = project / "runs" / job["run_id"]
        state = read_json(run / "state.json")
        if state["status"] != "complete" or digest_file(run / "last.pth") != state["checkpoint_sha256"]:
            raise ValueError(f"Benchmark requires an unchanged completed checkpoint: {job['run_id']}")
        checkpoint = load_checkpoint(run / "last.pth", "cpu")
        model = build_model(checkpoint["model_config"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        measurements = []
        with torch.inference_mode():
            for batch in (1, 16):
                x = torch.randn(batch, 3, settings["image_size"], settings["image_size"], device=device)
                for repeat in range(repeats):
                    for _ in range(warmup):
                        model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    timings = []
                    for _ in range(iterations):
                        began = time.perf_counter()
                        model(x)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        timings.append((time.perf_counter() - began) * 1000)
                    measurements.append(dict(batch=batch, repeat=repeat, median_ms=float(np.median(timings)),
                        p90_ms=float(np.percentile(timings, 90)), images_per_second=batch * 1000 / float(np.mean(timings)),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None))
        output = run / "benchmarks" / f"{device.type}_{time.time_ns()}.json"
        atomic_json(output, dict(environment=environment(device), warmup=warmup, iterations=iterations,
            repeats=repeats, measurements=measurements, includes_preprocessing=False,
            checkpoint_sha256=digest_file(run / "last.pth")))
        print(f"Benchmark saved: {output}")


def match_budget(project, metric="parameters", seeds=(42,), cpu=False):
    """Select cost-nearest candidates before training, without inspecting accuracy."""
    from experiment_suite import presets, add_jobs
    project, settings, frozen, plan = load_project(project)
    if (project / "final_registration.json").exists():
        raise ValueError("Final comparisons already frozen")
    if source_fingerprint() != settings["source_fingerprint"]:
        raise ValueError("Platform code changed since initialization")
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    choices = presets(settings, "budget")
    costs = {name: parameter_cost(build_model(config).to(device), settings["image_size"], device)
             for name, config in choices.items()}
    tolerance = .05 if metric == "parameters" else .10
    selected, comparisons = set(), []
    for target in ("G_C8", "G_C16", "G_C32"):
        value = costs[target][metric]
        candidate = min((name for name in costs if name.startswith("Q_")),
                        key=lambda name: (abs(costs[name][metric] - value), name))
        gap = abs(costs[candidate][metric] - value) / value
        matched = gap <= tolerance
        comparisons.append(dict(target=target, candidate=candidate, target_cost=value,
            candidate_cost=costs[candidate][metric], relative_gap=gap, matched=matched))
        selected.add(target)
        if matched:
            selected.add(candidate)
    record = dict(metric=metric, tolerance=tolerance, comparisons=comparisons, costs=costs,
                  note="Conv/Linear MAC matching is partial, not a complete FLOP match")
    atomic_json(project / f"budget_selection_{metric}.json", record)
    new = add_jobs(project, "budget", seeds, sorted(selected))
    print(f"Budget matching: added {len(new)} runs; unmatched controls are not scheduled")
    for comparison in comparisons:
        print(comparison)
