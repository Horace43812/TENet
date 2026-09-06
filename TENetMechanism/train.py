"""Train an independent TENet on the same ImageFolder layout as ViTPruning."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from common import seed_all, save_json, build_data, manifest, evaluate, load_checkpoint
from model import TENet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = p.add_argument_group("1. Dataset and subset")
    data.add_argument("--dataset", choices=["imagefolder", "fake"], default="imagefolder", help="Dataset type")
    data.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "dataset"), help="Dataset root containing train/ and val/")
    for name, default, help_text in [
        ("class-start", 0, "First class index in sorted class names"),
        ("class-count", 50, "Number of classes; 0 selects all remaining classes"),
        ("train-per-class", 500, "Training images per class; 0 uses all"),
        ("val-per-class", 0, "Validation images per class; 0 uses all"),
        ("image-size", 64, "Input image side length")]:
        data.add_argument("--" + name, type=int, default=default, help=help_text)
    data.add_argument("--no-aug", action="store_true", help="Use deterministic validation preprocessing for training")

    model = p.add_argument_group("2. TENet and classifier")
    for name, default, help_text in [
        ("layers", 3, "Number of graph layers; each halves spatial dimensions"),
        ("channels", 3, "Initial channels; values other than 3 enable a 1x1 stem"),
        ("alpha", 8, "Channel increment per layer; 0 keeps constant width"),
        ("head-grid", 4, "Classifier pooling grid side length"),
        ("head-hidden", 256, "Classifier hidden width")]:
        model.add_argument("--" + name, type=int, default=default, help=help_text)
    model.add_argument("--dropout", type=float, default=0.1, help="Classifier dropout")
    model.add_argument("--readout", choices=["topology", "content", "both"], default="topology", help="Feature stream used by the classifier")

    branches = p.add_argument_group("3. Graph branches")
    for name, default, help_text in [
        ("common-classes-only", True, "selection from classes shared by train and val"),
        ("learnable-phi", True, "learnable edge mapping"),
        ("trace", True, "trace branch"),
        ("edge", True, "edge feature enhancement"),
        ("directed", False, "directed differences"),
        ("cross", True, "cross-stream guidance"),
        ("chebyshev", False, "Laplacian order mixing"),
        ("spatial", False, "spatial bridge")]:
        group = (data if name == "common-classes-only" else branches).add_mutually_exclusive_group()
        group.add_argument("--" + name, dest=name.replace("-", "_"), action="store_true", help="Enable " + help_text)
        group.add_argument("--no-" + name, dest=name.replace("-", "_"), action="store_false", help="Disable " + help_text)
        p.set_defaults(**{name.replace("-", "_"): default})

    training = p.add_argument_group("4. Training and device")
    for name, default, help_text in [
        ("epochs", 100, "Total training epochs"), ("batch-size", 16, "Batch size"),
        ("workers", 4, "Data loader workers"), ("seed", 42, "Training and subset sampling seed")]:
        training.add_argument("--" + name, type=int, default=default, help=help_text)
    for name, default, help_text in [
        ("lr", 3e-4, "Initial learning rate"), ("weight-decay", 0.05, "Weight decay"),
        ("label-smoothing", 0.1, "Label smoothing")]:
        training.add_argument("--" + name, type=float, default=default, help=help_text)
    training.add_argument("--cpu", action="store_true", help="Force CPU execution")

    output = p.add_argument_group("5. Output and resume")
    output.add_argument("--output-dir", help="Output directory; defaults to runs/<UTC timestamp>")
    output.add_argument("--resume", help="Checkpoint path for resuming training")

    fake = p.add_argument_group("6. Fake-data smoke test")
    for name, default in [("num-classes", 10), ("fake-train-size", 64), ("fake-eval-size", 32)]:
        fake.add_argument("--" + name, type=int, default=default, help="Used only by the fake dataset")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.fake_train_size, args.fake_eval_size) < 1:
        p.error("epochs, batch size and fake sizes must be positive")
    if min(args.class_start, args.class_count, args.train_per_class, args.val_per_class, args.workers) < 0:
        p.error("subset limits and workers must be nonnegative")
    if not 0 <= args.label_smoothing < 1 or not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        p.error("invalid optimization parameters")
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = build_data(args)
    samples = manifest(data)
    config = {key: getattr(args, key) for key in ["image_size", "channels", "layers", "alpha", "head_grid",
        "head_hidden", "dropout", "readout", "learnable_phi", "trace", "edge", "directed", "cross", "chebyshev", "spatial"]}
    config["num_classes"] = data.num_classes
    model = TENet(**config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Device: {device} | Parameters: {parameter_count:,} | Trainable: {trainable_count:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    output = Path(args.output_dir) if args.output_dir else (Path(args.resume).resolve().parent if args.resume else
        Path(__file__).resolve().parent / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    history, start, best = [], 1, -1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        if checkpoint["model_config"] != config or checkpoint["manifest"] != samples:
            raise ValueError("Resume model and dataset manifest must match")
        for key in ["epochs", "seed", "batch_size", "lr", "weight_decay", "label_smoothing", "no_aug"]:
            if checkpoint["args"][key] != getattr(args, key):
                raise ValueError(f"Resume requires original {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start, best = checkpoint["history"], checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
    args.output_dir = str(output.resolve())
    save_json(output / "config.json", {"args": vars(args), "model_config": config})
    save_json(output / "manifest.json", samples)
    print(f"Output: {output.resolve()} | train/val={data.train_samples}/{data.eval_samples} | classes={data.num_classes}", flush=True)
    for epoch in range(start, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train = evaluate(model, data.train_loader, device, optimizer, args.label_smoothing)
        val = evaluate(model, data.eval_loader, device, smoothing=args.label_smoothing)
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "train": train, "val": val})
        improved = val["accuracy"] > best
        best = max(best, val["accuracy"])
        checkpoint = {"format_version": 1, "epoch": epoch, "model_config": config, "args": vars(args),
            "manifest": samples, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "best_accuracy": best}
        torch.save(checkpoint, output / "last.pth")
        if improved:
            torch.save(checkpoint, output / "best.pth")
        save_json(output / "metrics.json", history)
        print(f"Epoch {epoch:03d}/{args.epochs} | lr={lr:.3e} | "
              f"train loss={train['loss']:.4f} acc={train['accuracy']:.2f}% | "
              f"val loss={val['loss']:.4f} acc={val['accuracy']:.2f}% | best={best:.2f}%", flush=True)
    print(f"Checkpoint: {(output / 'last.pth').resolve()}", flush=True)


if __name__ == "__main__":
    main()
