"""Independent tiny MNIST experiment, with epoch-boundary resume."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from tiny_v1 import TinyMNIST


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        progress = tqdm(loader, desc="train" if optimizer is not None else "eval")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; last complete epoch remains saved")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count += labels.numel()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            progress.set_postfix(loss=loss_sum / count, acc=f"{100 * correct / count:.2f}%")
    return {"loss": loss_sum / count, "accuracy_pct": 100 * correct / count, "samples": count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normalization", choices=("raw", "centered", "standardized"), default="raw")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Code/data")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Synthetic data only; no MNIST download")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.lr) <= 0 or min(args.seed, args.workers, args.weight_decay) < 0:
        p.error("Invalid training arguments")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = args.output_dir or Path(__file__).resolve().parent / "runs" / f"{args.normalization}_s{args.seed}"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("output_dir", "resume", "workers", "cpu")}
    config["data_dir"] = str(args.data_dir.resolve())
    config["architecture"] = "tiny_g_1_2_3_linear10_v1"
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            p.error("Resume requires the original model/data/epochs/seed settings")
    elif out.exists() and any(out.iterdir()):
        p.error("Output is not empty; use --resume or another --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
    if args.smoke:
        train = datasets.FakeData(32, (1, 28, 28), 10, transform, random_offset=0)
        dev = datasets.FakeData(16, (1, 28, 28), 10, transform, random_offset=100)
        test = datasets.FakeData(16, (1, 28, 28), 10, transform, random_offset=200)
    else:
        source = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        # Fixed stratified split: 500 per class in dev, independent of training seed.
        generator = torch.Generator().manual_seed(20260913)
        train_ids, dev_ids = [], []
        for label in range(10):
            ids = (source.targets == label).nonzero().flatten()
            ids = ids[torch.randperm(len(ids), generator=generator)].tolist()
            dev_ids.extend(ids[:500])
            train_ids.extend(ids[500:])
        train, dev = Subset(source, train_ids), Subset(source, dev_ids)
        test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        write_json(out / "split.json", {"train": train_ids, "dev": dev_ids, "split_seed": 20260913})
    shuffle = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=shuffle, num_workers=args.workers)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, num_workers=args.workers)
    model = TinyMNIST(args.normalization).to(device)
    parameters = sum(v.numel() for v in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    history, start, best = [], 0, -1
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start, best = checkpoint["history"], checkpoint["epoch"], checkpoint["best_dev"]
        torch.set_rng_state(checkpoint["rng"])
        shuffle.set_state(checkpoint["shuffle_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    write_json(out / "config.json", config)
    print(f"Device: {device} | Parameters: {parameters:,} | Normalization: {args.normalization}", flush=True)
    print(f"Output: {out.resolve()} | Train/dev/test: {len(train)}/{len(dev)}/{len(test)}", flush=True)
    for index in range(start, args.epochs):
        begin = time.perf_counter()
        training = epoch(model, train_loader, device, optimizer)
        validation = epoch(model, dev_loader, device)
        history.append({"epoch": index + 1, "lr": optimizer.param_groups[0]["lr"],
                        "train": training, "dev": validation, "seconds": time.perf_counter() - begin})
        improved = validation["accuracy_pct"] > best
        best = max(best, validation["accuracy_pct"])
        scheduler.step()
        state = {"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": index + 1, "history": history,
                 "best_dev": best, "rng": torch.get_rng_state(), "shuffle_rng": shuffle.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        if improved:
            save_checkpoint(out / "best.pth", state)
        save_checkpoint(out / "last.pth", state)
        write_json(out / "metrics.json", history)
        print(f"Epoch {index + 1}/{args.epochs}: train={training['accuracy_pct']:.2f}% dev={validation['accuracy_pct']:.2f}%", flush=True)
    # Test only the last checkpoint after all epochs; never use test to select epochs.
    if not (out / "result.json").exists():
        testing = epoch(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.workers), device)
        write_json(out / "result.json", {"parameters": parameters, "normalization": args.normalization,
            "seed": args.seed, "epoch": args.epochs, "checkpoint": "last.pth", "test": testing,
            "synthetic": args.smoke, "torch": torch.__version__, "device": str(device)})
    print((out / "result.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
