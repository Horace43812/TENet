"""One-shot probe runner: train (or load) a TENet, then measure information flow.

Usage (from the Code/ directory):

    python -m analysis.run_probe --dataset mnist  --alpha 1 --epochs 5
    python -m analysis.run_probe --dataset cifar10 --alpha 8 --epochs 10
    python -m analysis.run_probe --dataset cifar10 --alpha 8 --checkpoint path.pt --no-train

Outputs land under ../docs/90_results/phase1_probe/<run_id>/:
    metrics.json   - raw numbers per layer
    summary.md     - human-readable table + config dump
    model.pt       - trained model state (unless --no-save-model)
    phi_layer*.npy - phi heatmap arrays for each layer (if available)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow `python -m analysis.run_probe` from the Code/ directory.
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis.cka import linear_cka
from analysis.effective_rank import effective_rank_per_channel
from analysis.hooks import ActivationCollector
from analysis.linear_probe import linear_probe_per_layer
from analysis.phi_visualize import sample_phi_grid


# --------------------------------------------------------------------------- #
# Config / CLI                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ProbeConfig:
    dataset: str = "mnist"
    alpha: int = 1
    num_layers: int = 2
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    probe_batches: int = 8            # how many val batches to collect for ER / CKA
    probe_epochs: int = 30
    probe_lr: float = 1e-2
    use_trace_branch: bool = True
    use_edge_enhance: bool = True
    use_cross_guidance: bool = True
    cpu: bool = False
    checkpoint: str = ""
    no_train: bool = False
    no_save_model: bool = False
    out_root: str = ""
    run_id: str = ""
    data_dir: str = "./data"
    max_train_batches: int = 0    # 0 = full epoch; >0 caps batches per epoch (for CPU runs)
    log_every: int = 0            # 0 = silent; >0 prints mid-epoch progress every N batches


def build_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description="TENet information-flow probe")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--no-trace-branch", action="store_true")
    parser.add_argument("--no-edge-enhance", action="store_true")
    parser.add_argument("--no-cross-guidance", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()

    cfg = ProbeConfig(
        dataset=args.dataset,
        alpha=args.alpha,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        probe_batches=args.probe_batches,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        use_trace_branch=not args.no_trace_branch,
        use_edge_enhance=not args.no_edge_enhance,
        use_cross_guidance=not args.no_cross_guidance,
        cpu=args.cpu,
        checkpoint=args.checkpoint,
        no_train=args.no_train,
        no_save_model=args.no_save_model,
        out_root=args.out_root,
        run_id=args.run_id,
        data_dir=args.data_dir,
        max_train_batches=args.max_train_batches,
        log_every=args.log_every,
    )
    return cfg


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def build_loaders(cfg: ProbeConfig):
    if cfg.dataset == "mnist":
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=tf)
        test = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=tf)
        return train, test, 10
    if cfg.dataset == "cifar10":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
        train = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=test_tf)
        return train, test, 10
    raise ValueError(f"unknown dataset {cfg.dataset}")


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def build_model(cfg: ProbeConfig, num_classes: int) -> nn.Module:
    if cfg.dataset == "mnist":
        from model_mnist28 import MNIST_TENet_Current
        return MNIST_TENet_Current(
            alpha=cfg.alpha,
            use_trace_branch=cfg.use_trace_branch,
            use_edge_enhance=cfg.use_edge_enhance,
            use_cross_stream_guidance=cfg.use_cross_guidance,
            use_directed_flow=False,
            use_chebyshev=False,
            use_spatial_bridge=False,
        )
    if cfg.dataset == "cifar10":
        from model_rgb32 import TENet_RGB32
        return TENet_RGB32(
            num_classes=num_classes,
            num_layers=cfg.num_layers,
            input_size=32,
            alpha=cfg.alpha,
            use_trace_branch=cfg.use_trace_branch,
            use_edge_enhance=cfg.use_edge_enhance,
            use_cross_stream_guidance=cfg.use_cross_guidance,
            use_directed_flow=False,
            use_chebyshev=False,
            use_spatial_bridge=False,
        )
    raise ValueError(cfg.dataset)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: ProbeConfig,
    device: torch.device,
) -> Dict[str, float]:
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            if cfg.max_train_batches and batch_idx >= cfg.max_train_batches:
                break
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
            if cfg.log_every and (batch_idx + 1) % cfg.log_every == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"  [ep{epoch:02d} b{batch_idx+1:04d}] "
                    f"loss={running/total:.4f} acc={correct/total*100:.2f}% "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
        train_loss = running / total
        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc}
        )
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%"
        )
    return {
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "history": history,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


# --------------------------------------------------------------------------- #
# Probing                                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_probe_activations(
    model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int
):
    """Pull `max_batches` worth of validation samples and return both raw inputs
    and per-layer outputs. All tensors land on CPU."""
    model.eval()
    inputs: List[torch.Tensor] = []
    per_layer: Optional[List[List[torch.Tensor]]] = None
    names: List[str] = []
    for batch_idx, (x, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        collector = ActivationCollector(model, detach=True, to_cpu=True)
        with collector:
            _ = model(x)
        recs = collector.records
        if per_layer is None:
            names = [r["name"] for r in recs]
            per_layer = [[] for _ in recs]
        for idx, rec in enumerate(recs):
            per_layer[idx].append(rec["topo"])
        inputs.append(x.detach().cpu())
    if per_layer is None:
        raise RuntimeError("loader was empty")
    inputs_cat = torch.cat(inputs, dim=0)
    layers_cat = [torch.cat(parts, dim=0) for parts in per_layer]
    return inputs_cat, layers_cat, names


def compute_metrics(
    inputs: torch.Tensor, layers: List[torch.Tensor], names: List[str]
) -> List[Dict]:
    """Per-layer effective rank + CKA(input, layer) + CKA(layer_l, layer_{l+1})."""
    metrics = []
    # Pre-flatten input for CKA — treat each sample as a single row so tensors
    # with mismatched spatial dims can still be compared.
    n_samples = inputs.shape[0]
    inputs_flat = inputs.reshape(n_samples, -1)
    layers_flat = [feat.reshape(feat.shape[0], -1) for feat in layers]
    for idx, feat in enumerate(layers):
        er = effective_rank_per_channel(feat)
        # CKA with input — use a balanced subsample if very large
        sub = min(n_samples, 2048)
        cka_in = linear_cka(inputs_flat[:sub], layers_flat[idx][:sub])
        cka_next = None
        if idx + 1 < len(layers):
            cka_next = linear_cka(layers_flat[idx][:sub], layers_flat[idx + 1][:sub])
        metrics.append(
            {
                "name": names[idx],
                "effective_rank": er["effective_rank"],
                "num_channels": er["num_channels"],
                "er_ratio": er["ratio"],
                "cka_input": cka_in,
                "cka_next": cka_next,
            }
        )
    return metrics


def collect_phi_grids(model: nn.Module) -> Dict[str, torch.Tensor]:
    from analysis.hooks import iter_graphified_layers
    out = {}
    for idx, (name, layer) in enumerate(iter_graphified_layers(model)):
        if getattr(layer, "phi", None) is None:
            continue
        try:
            grid = sample_phi_grid(model, layer_index=idx, channel=0)
            out[name] = grid
        except Exception as exc:
            print(f"[phi] skip {name}: {exc}")
    return out


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def write_report(
    out_dir: Path,
    cfg: ProbeConfig,
    train_stats: Optional[Dict],
    metrics: List[Dict],
    probe_results: List[Dict],
    num_params: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "num_params": num_params,
        "train_stats": train_stats,
        "layer_metrics": metrics,
        "linear_probe": probe_results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    lines: List[str] = []
    lines.append(f"# Probe run `{cfg.run_id or out_dir.name}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append(f"- **#params**: {num_params:,}")
    if train_stats:
        lines.append(f"- **final train acc**: {train_stats['final_train_acc']*100:.2f}%")
        lines.append(f"- **final val acc**:   {train_stats['final_val_acc']*100:.2f}%")
    lines.append("")
    lines.append("## Per-layer information-flow metrics")
    lines.append("")
    lines.append("| layer | C | eff.rank | rank/C | CKA(input, layer) | CKA(layer, next) | probe val acc |")
    lines.append("|---|---|---|---|---|---|---|")
    probe_by_name = {p["name"]: p for p in probe_results}
    for m in metrics:
        probe = probe_by_name.get(m["name"], {})
        probe_acc = probe.get("val_acc")
        probe_str = f"{probe_acc*100:.2f}%" if probe_acc is not None else "—"
        cka_next = m.get("cka_next")
        cka_next_str = f"{cka_next:.4f}" if cka_next is not None else "—"
        lines.append(
            f"| `{m['name']}` "
            f"| {m['num_channels']} "
            f"| {m['effective_rank']:.3f} "
            f"| {m['er_ratio']:.3f} "
            f"| {m['cka_input']:.4f} "
            f"| {cka_next_str} "
            f"| {probe_str} |"
        )
    lines.append("")
    if train_stats and "history" in train_stats:
        lines.append("## Training history")
        lines.append("")
        lines.append("| epoch | train_loss | train_acc | val_acc |")
        lines.append("|---|---|---|---|")
        for h in train_stats["history"]:
            lines.append(
                f"| {h['epoch']} | {h['train_loss']:.4f} | "
                f"{h['train_acc']*100:.2f}% | {h['val_acc']*100:.2f}% |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    cfg = build_args()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    print(f"[device] {device}")

    train_set, test_set, num_classes = build_loaders(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    model = build_model(cfg, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[model] #params={num_params:,}")

    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        state = torch.load(cfg.checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"[ckpt] loaded {cfg.checkpoint}")

    train_stats = None
    if not cfg.no_train:
        train_stats = quick_train(model, train_loader, test_loader, cfg, device)

    print("[probe] collecting activations")
    inputs, layers, names = collect_probe_activations(model, test_loader, device, cfg.probe_batches)
    print(f"[probe] #samples={inputs.shape[0]}, #layers={len(names)}")

    metrics = compute_metrics(inputs, layers, names)
    for m in metrics:
        print(
            f"[er] {m['name']:>20s} C={m['num_channels']:<4d} "
            f"rank={m['effective_rank']:.3f} ratio={m['er_ratio']:.3f} "
            f"cka_in={m['cka_input']:.4f}"
        )

    print("[probe] running per-layer linear probe")
    probe_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    probe_results = linear_probe_per_layer(
        model, probe_loader, num_classes, device=device,
        probe_epochs=cfg.probe_epochs, probe_lr=cfg.probe_lr,
    )
    for p in probe_results:
        print(f"[probe] {p['name']:>20s} dim={p['dim']:<6d} val_acc={p['val_acc']*100:.2f}%")

    # Output dir
    if cfg.out_root:
        root = Path(cfg.out_root)
    else:
        root = CODE_DIR.parent / "docs" / "90_results" / "phase1_probe"
    if not cfg.run_id:
        ts = time.strftime("%Y%m%d_%H%M%S")
        modules = []
        if cfg.use_trace_branch: modules.append("tr")
        if cfg.use_edge_enhance: modules.append("eg")
        if cfg.use_cross_guidance: modules.append("cg")
        mod_tag = "+".join(modules) if modules else "none"
        cfg.run_id = (
            f"{ts}_{cfg.dataset}_a{cfg.alpha}_L{cfg.num_layers}_{mod_tag}"
        )
    out_dir = root / cfg.run_id

    write_report(out_dir, cfg, train_stats, metrics, probe_results, num_params)

    phi_grids = collect_phi_grids(model)
    for name, grid in phi_grids.items():
        safe = name.replace(".", "_")
        torch.save(grid, out_dir / f"phi_{safe}.pt")

    if not cfg.no_save_model:
        torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"[done] report written to {out_dir}")


if __name__ == "__main__":
    main()
