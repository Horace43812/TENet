"""Tool 3: Input-gradient saliency for cross-subtree leakage diagnosis.

Given a target layer and a target spatial location (i, j), compute
|d ||f_ij||^2 / d x_pq| heatmap over the input pixels. Used to verify whether
information flow respects the quadtree structure or leaks across subtrees.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .hooks import iter_graphified_layers


def input_grad_saliency(
    model: nn.Module,
    x: torch.Tensor,
    layer_index: int,
    target_ij: Optional[tuple] = None,
    stream: str = "topo",
) -> torch.Tensor:
    """Compute input-gradient saliency map.

    Args:
        model: TENet model (already in eval mode).
        x: input tensor of shape (1, C, H, W). Will be cloned and require_grad-ed.
        layer_index: which GraphifiedLayer to target (0-based in forward order).
        target_ij: (i, j) coordinate within that layer's spatial map. Defaults
                   to the center.
        stream: "topo" or "content" (topo by default).

    Returns:
        Saliency map of shape (C, H, W) on CPU.
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    layers = list(iter_graphified_layers(model))
    if not layers:
        raise RuntimeError("No GraphifiedLayer found in model")
    if not (0 <= layer_index < len(layers)):
        raise IndexError(f"layer_index {layer_index} out of range (have {len(layers)})")

    target_name, _ = layers[layer_index]
    captured = {}

    def hook(module, inputs, output):
        out = output[0] if stream == "topo" else output[1]
        captured["out"] = out

    handle = layers[layer_index][1].register_forward_hook(hook)
    try:
        _ = model(x)
    finally:
        handle.remove()

    feat = captured["out"]  # (1, C', H', W')
    if feat.dim() != 4:
        raise RuntimeError(f"Expected 4D feature, got {tuple(feat.shape)}")
    _, _, h, w = feat.shape
    i, j = target_ij if target_ij is not None else (h // 2, w // 2)
    target_scalar = feat[0, :, i, j].pow(2).sum()
    grad = torch.autograd.grad(target_scalar, x, retain_graph=False)[0]
    return grad.detach().abs().squeeze(0).cpu()
