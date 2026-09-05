"""End-to-end training entry point for the independent ViT pruning platform."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn, optim

from datasets import build_dataloaders
from model_vit import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a prunable ViT",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["fake", "cifar10", "imagefolder"], default="fake")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "dataset"),
                        help="dataset root containing train/ and val/; unused by fake")
    parser.add_argument("--class-start", type=int, default=0, help="first class index in sorted folder names (imagefolder)")
    parser.add_argument("--class-count", type=int, default=0, help="number of classes; 0 = all remaining (imagefolder)")
    parser.add_argument("--common-classes-only", action="store_true",
                        help="select class range from sorted train/val intersection (imagefolder)")
    parser.add_argument("--train-per-class", type=int, default=0, help="training images per class; 0 = all (imagefolder)")
    parser.add_argument("--val-per-class", type=int, default=0, help="validation images per class; 0 = all (imagefolder)")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10, help="used by fake; inferred by imagefolder")
    parser.add_argument("--fake-train-size", type=int, default=256)
    parser.add_argument("--fake-eval-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pruner", choices=["identity", "magnitude", "linear", "tenet_local"], default="magnitude")
    parser.add_argument("--prune-keep-ratio", type=float, default=0.5)
    parser.add_argument("--prune-at-block", type=int, default=None)
    parser.add_argument("--resume", default="", help="path to a platform checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
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
    return args


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
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
    return {"loss": loss_sum / total, "accuracy": 100.0 * correct / total}


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    args = parse_args()
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
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,} | "
          f"pruner: {args.pruner} after block {model.prune_at_block} | keep: {args.prune_keep_ratio}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, data.train_loader, criterion, device, optimizer)
        eval_metrics = run_epoch(model, data.eval_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "eval": eval_metrics}
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
        print(f"Epoch {epoch:03d}/{args.epochs} | train {train_metrics['loss']:.4f} / {train_metrics['accuracy']:.2f}% | "
              f"eval {eval_metrics['loss']:.4f} / {eval_metrics['accuracy']:.2f}% | best {best_accuracy:.2f}%")
    print(f"Artifacts: {best_path}, {last_path}, {metrics_path}")


if __name__ == "__main__":
    main()
