"""End-to-end training entry point for the independent ViT pruning platform."""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from datasets import build_dataloaders
from model_vit import VisionTransformer
from results import append_result


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a prunable ViT",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["fake", "cifar10", "imagefolder"], default="imagefolder")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "dataset"),
                        help="dataset root containing train/ and val/; unused by fake")
    parser.add_argument("--class-start", type=int, default=0, help="first class index in sorted folder names (imagefolder)")
    parser.add_argument("--class-count", type=int, default=None, help="number of classes; default: 20 for imagefolder; 0 = all remaining")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--common-classes-only", action="store_true", default=None,
                        help="select class range from sorted train/val intersection (imagefolder)")
    selection.add_argument("--no-common-classes-only", dest="common_classes_only", action="store_false", default=None,
                           help="require selected training classes to exist in val")
    parser.add_argument("--train-per-class", type=int, default=None, help="training images per class; default: 500 for imagefolder; 0 = all")
    parser.add_argument("--val-per-class", type=int, default=None, help="validation images per class; default: all for imagefolder; 0 = all")
    parser.add_argument("--output-dir", default=None, help="new runs: checkpoints/<pruner>/<timestamp>_seed<seed>; resume: checkpoint directory")
    parser.add_argument("--results-file", default=str(Path(__file__).resolve().parent / "checkpoints" / "training_results.jsonl"),
                        help="append one JSON record per completed training run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10, help="used by fake; inferred by imagefolder")
    parser.add_argument("--fake-train-size", type=int, default=256)
    parser.add_argument("--fake-eval-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pruner", choices=["identity", "magnitude", "linear", "tenet_local"], default="magnitude")
    parser.add_argument("--prune-keep-ratio", type=float, default=0.5)
    parser.add_argument("--prune-at-block", type=int, default=6)
    parser.add_argument("--resume", default="", help="path to a platform checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    for name, default in (("class_count", 20), ("train_per_class", 500), ("val_per_class", 0),
                          ("common_classes_only", True)):
        if getattr(args, name) is None:
            setattr(args, name, default if args.dataset == "imagefolder" else 0)
    if min(args.class_start, args.class_count, args.train_per_class, args.val_per_class, args.workers) < 0:
        parser.error("subset limits and workers must be nonnegative")
    if args.dataset != "imagefolder" and any((args.class_start, args.class_count, args.train_per_class,
                                              args.val_per_class, args.common_classes_only)):
        parser.error("class and per-class selection options require --dataset imagefolder")
    if min(args.epochs, args.batch_size, args.fake_train_size, args.fake_eval_size) <= 0:
        parser.error("epochs, batch-size, and fake dataset sizes must be positive")
    if min(args.image_size, args.patch_size, args.embed_dim, args.depth, args.num_heads) <= 0:
        parser.error("image-size, patch-size, embed-dim, depth, and num-heads must be positive")
    if args.image_size % args.patch_size:
        parser.error("image-size must be divisible by patch-size")
    if args.embed_dim % args.num_heads:
        parser.error("embed-dim must be divisible by num-heads")
    if not 0.0 < args.prune_keep_ratio <= 1.0:
        parser.error("prune-keep-ratio must be in (0, 1]")
    if args.prune_at_block is not None and not 1 <= args.prune_at_block <= args.depth:
        parser.error("prune-at-block must be between 1 and depth")
    if args.pruner == "tenet_local" and (args.image_size // args.patch_size) % 2:
        parser.error("tenet_local requires an even patch grid side length")
    if args.output_dir is None:
        args.output_dir = str(Path(args.resume).resolve().parent) if args.resume else str(
            Path(__file__).resolve().parent / "checkpoints" / args.pruner /
            (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_seed{args.seed}"))
    return args


def run_epoch(model, loader, criterion, device, optimizer=None, progress_desc=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context, tqdm(loader, desc=progress_desc, disable=progress_desc is None,
                       file=sys.stdout, dynamic_ncols=True, mininterval=1.0, unit="batch", ascii=True) as progress:
        for images, labels in progress:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss_sum / total:.4f}", acc=f"{100 * correct / total:.2f}%", refresh=False)
    return {"loss": loss_sum / total, "accuracy": 100.0 * correct / total}


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = build_dataloaders(args.dataset, args.data_dir, args.image_size, args.batch_size, args.workers,
                             not args.no_aug, args.num_classes, args.fake_train_size, args.fake_eval_size,
                             args.class_start, args.class_count, args.train_per_class, args.val_per_class, args.seed,
                             args.common_classes_only)
    data_config = {"dataset": args.dataset, "data_dir": str(Path(args.data_dir).resolve()),
                   "class_names": data.class_names, "class_to_idx": {name: i for i, name in enumerate(data.class_names)},
                   "train_samples": data.train_samples, "val_samples": data.eval_samples,
                   "class_start": args.class_start, "class_count": args.class_count,
                   "train_per_class": args.train_per_class, "val_per_class": args.val_per_class, "seed": args.seed}
    model_config = dict(image_size=args.image_size, patch_size=args.patch_size, in_channels=3,
                        num_classes=data.num_classes, embed_dim=args.embed_dim, depth=args.depth,
                        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
                        pruner_name=args.pruner, prune_keep_ratio=args.prune_keep_ratio,
                        prune_at_block=args.prune_at_block)
    model = VisionTransformer(**model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    output_dir = Path(args.output_dir)
    if not args.resume and any((output_dir / name).exists() for name in
                               ("config.json", "metrics.json", "best.pth", "last.pth")):
        raise FileExistsError(f"Training artifacts already exist in {output_dir}; choose a new --output-dir or --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, best_path, last_path = output_dir / "metrics.json", output_dir / "best.pth", output_dir / "last.pth"
    start_epoch, best_accuracy, history = 1, -1.0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("model_config") != model_config:
            raise ValueError("Resume requires the same model configuration as the checkpoint")
        previous_data = checkpoint.get("data_config")
        if previous_data is not None and previous_data != data_config:
            raise ValueError("Resume requires the same dataset, class mapping, subset limits and seed")
        if checkpoint.get("args", {}).get("epochs", args.epochs) != args.epochs:
            raise ValueError("Resume requires the original --epochs total to preserve the learning-rate schedule")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint.get("best_accuracy", checkpoint.get("eval_accuracy", -1.0)))
        history = checkpoint.get("history", [])
    save_json(output_dir / "config.json", {"args": vars(args), "model_config": model_config, "data_config": data_config})
    print(f"Dataset: {args.dataset} | train/eval: {data.train_samples}/{data.eval_samples} | classes: {data.num_classes}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,} | "
          f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | "
          f"pruner: {args.pruner} after block {model.prune_at_block} | keep: {args.prune_keep_ratio}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(model, data.train_loader, criterion, device, optimizer,
                                  progress_desc=f"Epoch {epoch}/{args.epochs} [Train]")
        eval_metrics = run_epoch(model, data.eval_loader, criterion, device,
                                 progress_desc=f"Epoch {epoch}/{args.epochs} [Eval]")
        scheduler.step()
        record = {"epoch": epoch, "lr": epoch_lr, "train": train_metrics, "eval": eval_metrics}
        history.append(record)
        checkpoint = {"epoch": epoch, "args": vars(args), "model_config": model_config,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(), "eval_accuracy": eval_metrics["accuracy"],
                      "best_accuracy": max(best_accuracy, eval_metrics["accuracy"]), "history": history,
                      "data_config": data_config}
        torch.save(checkpoint, last_path)
        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            torch.save(checkpoint, best_path)
        save_json(metrics_path, {"args": vars(args), "model_config": model_config, "history": history})
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.2f}% | "
              f"eval loss={eval_metrics['loss']:.4f} acc={eval_metrics['accuracy']:.2f}% | best {best_accuracy:.2f}%", flush=True)
    print(f"Artifacts: {best_path}, {last_path}, {metrics_path}")
    if start_epoch <= args.epochs:
        append_result(args.results_file, args=vars(args), model_config=model_config, data_config=data_config,
                      history=history, started_at=started_at, elapsed_seconds=time.perf_counter() - start_time,
                      device=str(device), parameter_count=sum(p.numel() for p in model.parameters()),
                      start_epoch=start_epoch)
        print(f"Results appended: {Path(args.results_file).resolve()}")


if __name__ == "__main__":
    main()
