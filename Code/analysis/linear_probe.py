"""Tool 4: Per-layer Linear Probe.

Freezes the trained TENet, extracts each GraphifiedLayer's flattened topo
output, and fits a logistic-regression-style linear classifier on top to
estimate how much task-relevant information that layer carries.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .hooks import ActivationCollector


@torch.no_grad()
def _extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[torch.Tensor], torch.Tensor, List[str]]:
    """Run the model over the loader once, return per-layer flattened features."""
    model.eval()
    per_layer: List[List[torch.Tensor]] = []
    labels: List[torch.Tensor] = []
    names: List[str] = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        collector = ActivationCollector(model, detach=True, to_cpu=True)
        with collector:
            _ = model(x)
        records = collector.records

        if batch_idx == 0:
            names = [r["name"] for r in records]
            per_layer = [[] for _ in records]

        for idx, rec in enumerate(records):
            feat = rec["topo"]
            per_layer[idx].append(feat.reshape(feat.shape[0], -1))
        labels.append(y.clone())

    features = [torch.cat(parts, dim=0) for parts in per_layer]
    targets = torch.cat(labels, dim=0)
    return features, targets, names


def _train_linear_probe(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> float:
    feats = feats.float()
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    feats = (feats - mean) / std

    n = feats.shape[0]
    n_train = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    x_train, y_train = feats[train_idx].to(device), labels[train_idx].to(device)
    x_val, y_val = feats[val_idx].to(device), labels[val_idx].to(device)

    clf = nn.Linear(feats.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0.0
    for _ in range(epochs):
        clf.train()
        idx = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, idx.shape[0], batch_size):
            sel = idx[start : start + batch_size]
            logits = clf(x_train[sel])
            loss = F.cross_entropy(logits, y_train[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pred = clf(x_val).argmax(dim=1)
            acc = (pred == y_val).float().mean().item()
        best_val = max(best_val, acc)
    return best_val


def linear_probe_per_layer(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    probe_epochs: int = 30,
    probe_lr: float = 1e-2,
) -> List[Dict]:
    """Return a list of {name, dim, val_acc} dicts, one per GraphifiedLayer."""
    feats, labels, names = _extract_features(model, loader, device)
    results: List[Dict] = []
    for name, feat in zip(names, feats):
        acc = _train_linear_probe(
            feat,
            labels,
            num_classes=num_classes,
            epochs=probe_epochs,
            lr=probe_lr,
            device=device,
        )
        results.append({"name": name, "dim": int(feat.shape[1]), "val_acc": float(acc)})
    return results
