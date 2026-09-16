"""Independent tiny MNIST experiment, with epoch-boundary resume."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from tiny_v1 import TinyMNIST
from runtime import epoch, write_json, save_checkpoint, summaries, run_training
from records import RunRecord, DEFAULT_RECORDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normalization", choices=("raw", "centered", "standardized"), default="raw")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Code/data")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS)
    p.add_argument("--epochs", type=int, default=50)
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
              if k not in ("output_dir", "records_dir", "resume", "workers", "cpu")}
    config["data_dir"] = str(args.data_dir.resolve())
    config["architecture"] = "tiny_g_1_2_3_linear10_v1"
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            p.error("Resume requires the original model/data/epochs/seed settings")
    elif out.exists() and any(out.iterdir()):
        p.error("Output is not empty; use --resume or another --output-dir")
    with RunRecord(args.records_dir, out, config) as record:
        execute(args, config, checkpoint, out, device, record)


def execute(args, config, checkpoint, out, device, record):
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
    start = 0
    best_summary, last_summary = summaries(checkpoint, "dev")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["rng"])
        shuffle.set_state(checkpoint["shuffle_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    write_json(out / "config.json", config)
    print(f"Device: {device} | Parameters: {parameters:,} | Normalization: {args.normalization}", flush=True)
    print(f"Output: {out.resolve()} | Train/dev/test: {len(train)}/{len(dev)}/{len(test)}", flush=True)
    record.update(parameters=parameters, normalization=args.normalization,
                  model_config={'in_channels': 1, 'image_size': 28, 'num_classes': 10, 'layers': 2, 'alpha': 1},
                  resolved_model={'version': 'tiny_v1', 'normalization': args.normalization},
                  device=str(device), torch=torch.__version__, workers=args.workers, threads=4)
    run_training(model=model, loader=train_loader, evaluation=dev_loader,
                 test_loader=DataLoader(test, batch_size=args.batch_size, num_workers=args.workers),
                 optimizer=optimizer, scheduler=scheduler, shuffle=shuffle, device=device, out=out,
                 epochs=args.epochs, start=start, monitor='dev', selection_metric='accuracy_pct',
                 best_summary=best_summary, last_summary=last_summary, config=config, state_extra={},
                 record=record, epoch_fn=epoch)


if __name__ == "__main__":
    main()
