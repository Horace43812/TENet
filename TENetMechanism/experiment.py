"""Integrated TENet experiments: init -> run -> report -> confirm -> frozen final."""
import argparse
import importlib.util
import subprocess
import shutil
import sys
import time
from pathlib import Path

from experiment_data import atomic_json, fingerprint, prepare_manifest
from experiment_suite import BATCHES, add_jobs, load_project, project_lock, source_fingerprint, presets, validate_choices


DEFAULT_PROJECT = Path(__file__).resolve().parent / "experiments" / "main"
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "dataset"


def csv_names(value):
    names = [s.strip() for s in value.split(",") if s.strip()]
    if not names or len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("Provide distinct comma-separated values")
    return names


def csv_seeds(value):
    try:
        seeds = [int(s) for s in csv_names(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be integers") from error
    if min(seeds) < 0:
        raise argparse.ArgumentTypeError("Seeds must be nonnegative")
    return seeds


def initialize(args):
    project = args.project.resolve()
    if (project / "project.json").exists():
        raise ValueError("Project already exists; use run to resume, or choose a new --project")
    if any(p.name != ".lock" for p in project.iterdir()):
        raise ValueError("Use an empty project directory; existing artifacts will not be overwritten")
    if min(args.epochs, args.image_size, args.width, args.head_grid, args.head_hidden, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("Training and model dimensions must be positive")
    if args.image_size % 8 or args.head_grid > args.image_size // 8:
        raise ValueError("Core suite requires image size divisible by 8 and head-grid <= image-size/8")
    if args.batch_size < 2:
        raise ValueError("Training batch-size must be at least 2 for BatchNorm at small spatial sizes")
    preview = dict(width=args.width, num_classes=2, image_size=args.image_size,
                   head_grid=args.head_grid, head_hidden=args.head_hidden)
    validate_choices(preview, presets(preview, args.batch), args.models)
    print("Scanning images, checking duplicates and freezing train/dev/final manifests...", flush=True)
    manifest = prepare_manifest(args.data_dir, args.class_count, args.class_start, args.train_per_class,
        final_unseen=args.final_unseen, groups_file=args.groups_file)
    settings = dict(version=1, protocol="graphification_recursion_v1", data_dir=str(args.data_dir.resolve()),
        image_size=args.image_size, width=args.width, head_grid=args.head_grid, head_hidden=args.head_hidden,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, num_classes=len(manifest["class_names"]),
        manifest_fingerprint=manifest["fingerprint"], source_fingerprint=source_fingerprint(),
        preprocessing=dict(train="RandomResizedCrop(scale=.08..1,ratio=.75..1.333)+HorizontalFlip(p=.5)",
            evaluation="Resize(round(size/.875))+CenterCrop", interpolation="bilinear, torchvision defaults",
            mean=[.485, .456, .406], std=[.229, .224, .225]))
    settings["fingerprint"] = fingerprint(settings)
    atomic_json(project / "manifest.json", manifest)
    atomic_json(project / "project.json", settings)
    snapshot = project / "source_snapshot"
    snapshot.mkdir()
    for source in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    shutil.copy2(Path(__file__).with_name("requirements.txt"), snapshot / "requirements.txt")
    atomic_json(project / "plan.json", {"jobs": [], "fingerprint": fingerprint([])})
    jobs = add_jobs(project, args.batch, (42,), args.models)
    print(f"Prepared {len(jobs)} runs at {project}")
    print("Samples:", {k: len(v) for k, v in manifest["splits"].items()})
    print("Final status:", manifest["final_status"])
    from experiment_report import report_project
    print("Dashboard:", report_project(project))


def doctor():
    print("Python:", sys.executable, sys.version.split()[0])
    missing = []
    for name in ("torch", "torchvision", "numpy", "PIL", "matplotlib", "tqdm"):
        available = importlib.util.find_spec(name) is not None
        print(name, "OK" if available else "MISSING")
        if not available:
            missing.append(name)
    if missing:
        raise RuntimeError("Install dependencies in this interpreter: python -m pip install -r requirements.txt")
    import torch
    import torchvision
    from model import build_model
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_model(dict(architecture="research", num_classes=2, image_size=8, channels=3,
                             operators=["G", "Q", "G"], head_grid=1, head_hidden=8))
    model(torch.randn(2, 3, 8, 8)).sum().backward()
    print("Forward/backward: OK | CUDA:", torch.cuda.is_available(), "| TorchVision:", torchvision.__version__)


def smoke(project):
    import numpy as np
    from PIL import Image
    root = project.resolve()
    if root.exists():
        raise ValueError("Smoke project must not already exist")
    data = root / "synthetic_data"
    rng = np.random.default_rng(4321)
    for split in ("train", "val"):
        for label in (0, 1):
            folder = data / split / str(label)
            folder.mkdir(parents=True)
            for i in range(8 if split == "train" else 3):
                pixels = rng.integers(0, 30, (12, 12, 3), dtype=np.uint8)
                pixels[:, :, label * 2] += 180
                Image.fromarray(pixels).save(folder / f"{i}.png")
    destination = root / "project"
    init_args = parser().parse_args(["init", "--project", str(destination), "--data-dir", str(data),
        "--epochs", "1", "--image-size", "8", "--width", "3", "--head-grid", "1", "--head-hidden", "8",
        "--batch-size", "2", "--batch", "positions", "--models", "G,Q,GQQ"])
    with project_lock(destination):
        initialize(init_args)
        from experiment_runner import run_project, analyze_project, freeze_final, evaluate_final
        import torch
        torch.set_num_threads(1)
        run_project(destination, cpu=True)
        run_project(destination, cpu=True)  # verify completed-run skipping
        analyze_project(destination, models=["GQQ"], cpu=True, probe_epochs=1, probe_dim=8)
        freeze_final(destination, ["G", "Q"], [42])
        evaluate_final(destination, cpu=True)
        from experiment_report import report_project
        print("SMOKE PASS (synthetic pipeline check only):", report_project(destination, plots=True))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("start", "init", "plan", "run", "status", "check", "report", "analyze", "benchmark", "freeze-final", "final", "doctor", "smoke", "cost", "match-budget"):
        command = sub.add_parser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        command.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Experiment project directory")
        if name in ("init", "plan"):
            command.add_argument("--batch", choices=BATCHES if name == "plan" else BATCHES[:-1], default="core")
        if name in ("start", "init", "plan", "run", "analyze", "benchmark", "freeze-final", "cost"):
            command.add_argument("--models", type=csv_names, required=name == "freeze-final", help="Comma-separated model ids")
        if name in ("plan", "run", "analyze", "benchmark", "freeze-final", "cost", "match-budget"):
            command.add_argument("--seeds", type=csv_seeds, required=name == "freeze-final", help="Comma-separated training seeds")
        if name in ("start", "run", "analyze", "benchmark", "final", "cost", "match-budget"):
            command.add_argument("--cpu", action="store_true")
            command.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
        if name == "init":
            command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
            command.add_argument("--class-start", type=int, default=0)
            command.add_argument("--class-count", type=int, default=50)
            command.add_argument("--train-per-class", type=int, default=500)
            command.add_argument("--image-size", type=int, default=64)
            command.add_argument("--width", type=int, default=16)
            command.add_argument("--head-grid", type=int, default=4)
            command.add_argument("--head-hidden", type=int, default=256)
            command.add_argument("--epochs", type=int, default=100)
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--lr", type=float, default=3e-4)
            command.add_argument("--groups-file", type=Path, help="JSON mapping relative image path to subject/source group")
            command.add_argument("--final-unseen", action="store_true", help="Declare original val has never been used for tuning")
        if name == "start":
            command.add_argument("--data-dir", type=Path, help="For first initialization; defaults to sibling dataset")
        if name in ("start", "run"):
            command.add_argument("--workers", type=int, default=0, help="0 is the portable default")
            command.add_argument("--keep-going", action="store_true", help="Continue other runs after a failure")
            command.add_argument("--analyze", action="store_true", help="Also run layer probes and A-E after training")
        if name == "analyze":
            command.add_argument("--probe-epochs", type=int, default=30)
            command.add_argument("--probe-dim", type=int, default=256)
        if name == "benchmark":
            command.add_argument("--iterations", type=int, default=200)
            command.add_argument("--warmup", type=int, default=50)
            command.add_argument("--repeats", type=int, default=5)
        if name == "match-budget":
            command.add_argument("--metric", choices=["parameters", "conv_linear_macs"], default="parameters")
    return p


def menu():
    print("\nTENet 实验平台\n1. 初始化数据和核心实验\n2. 一键开始/继续实验\n3. 查看进度并生成报告\n4. 运行逐层分析\n5. 检查环境\n6. 小数据全流程自检\n0. 退出")
    choice = input("选择 [2]：").strip() or "2"
    if choice == "0":
        return
    commands = {"1": "init", "2": "start", "3": "report", "4": "analyze", "5": "doctor", "6": "smoke"}
    if choice not in commands:
        raise ValueError("请选择0–6")
    argv = [commands[choice]]
    if choice == "1":
        root = input(f"数据目录 [{DEFAULT_DATA}]：").strip().strip('"')
        if root:
            argv += ["--data-dir", root]
    if choice == "6":
        argv += ["--project", str(DEFAULT_PROJECT.parent / f"smoke_{time.time_ns()}")]
    main(argv)


def main(argv=None):
    if argv is None and len(sys.argv) == 1:
        if sys.stdin.isatty():
            return menu()
        parser().print_help()
        return
    args = parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "smoke":
        return smoke(args.project)
    if getattr(args, "threads", 1) < 1 or getattr(args, "workers", 0) < 0:
        raise ValueError("Threads must be positive and workers nonnegative")
    for name in ("probe_epochs", "probe_dim", "iterations", "repeats"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if getattr(args, "warmup", 0) < 0:
        raise ValueError("warmup must be nonnegative")
    if args.command == "status":
        # Pure read-only status must remain available while the training process holds the lock.
        from experiment_report import project_status
        for row in project_status(args.project):
            print(f"{row['run_id']:24} {row['status']:12} epoch={row['epoch']} dev={row['dev_accuracy']}")
        print("Dashboard:", args.project.resolve() / "report/index.html")
        return
    with project_lock(args.project):
        if args.command == "start":
            if not (args.project / "project.json").exists():
                initial = ["init", "--project", str(args.project), "--data-dir", str(args.data_dir or DEFAULT_DATA)]
                initialize(parser().parse_args(initial))
            elif args.data_dir:
                _, settings, _, _ = load_project(args.project)
                if args.data_dir.resolve() != Path(settings["data_dir"]).resolve():
                    raise ValueError("Existing project has another data directory; choose a new --project")
            args.seeds = None
            args.command = "run"
        if args.command == "init":
            return initialize(args)
        if args.command == "plan":
            seeds = args.seeds or ([43, 44] if args.batch == "confirm" else [42])
            new = add_jobs(args.project, args.batch, seeds, args.models)
            print(f"Added {len(new)} runs; duplicates skipped")
        elif args.command not in ("status", "report"):
            import torch
            torch.set_num_threads(getattr(args, "threads", 4))
            from experiment_runner import run_project, analyze_project, benchmark, freeze_final, evaluate_final, parameter_cost, selected_jobs, check_project
            if args.command == "run":
                run_project(args.project, args.cpu, args.workers, args.models, args.seeds, args.keep_going)
                if args.analyze:
                    analyze_project(args.project, args.models, args.seeds, args.cpu)
            elif args.command == "analyze":
                analyze_project(args.project, args.models, args.seeds, args.cpu, args.probe_epochs, args.probe_dim)
            elif args.command == "benchmark":
                benchmark(args.project, args.models, args.seeds, args.cpu, args.iterations, args.warmup, args.repeats)
            elif args.command == "freeze-final":
                freeze_final(args.project, args.models, args.seeds)
            elif args.command == "final":
                evaluate_final(args.project, args.cpu)
            elif args.command == "check":
                check_project(args.project)
            elif args.command == "match-budget":
                from experiment_runner import match_budget
                match_budget(args.project, args.metric, args.seeds or [42], args.cpu)
            elif args.command == "cost":
                project, settings, manifest, plan = load_project(args.project)
                from model import build_model
                device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
                for job in selected_jobs(plan, args.models, args.seeds):
                    model = build_model(job["model_config"]).to(device)
                    value = parameter_cost(model, settings["image_size"], device)
                    atomic_json(project / "runs" / job["run_id"] / "cost.json", value)
                    print(job["run_id"], value)
        from experiment_report import report_project
        report = report_project(args.project, plots=args.command == "report")
        if args.command == "status":
            from experiment_data import read_json
            for row in read_json(report.parent / "summary.json")["runs"]:
                print(f"{row['run_id']:24} {row['status']:12} epoch={row['epoch']} dev={row['dev_accuracy']}")
        print("Dashboard:", report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。再次运行相同命令会从已保存轮次继续。", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
