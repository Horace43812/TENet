"""Tool 5: Visualise the learned edge-strength function phi.

Sweeps (v_i, v_j) over a 2-D grid and records phi(v_i, v_j) for a chosen
output channel. Returns a numpy-friendly tensor that can be plotted by the
caller.
"""

from __future__ import annotations

from typing import Optional

import torch

from .hooks import iter_graphified_layers


@torch.no_grad()
def sample_phi_grid(
    model,
    layer_index: int,
    channel: int = 0,
    grid_size: int = 64,
    value_range: tuple = (0.0, 1.0),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a (grid_size, grid_size) tensor of phi(v_i, v_j) values."""
    layers = list(iter_graphified_layers(model))
    if not layers:
        raise RuntimeError("No GraphifiedLayer found in model")
    if not (0 <= layer_index < len(layers)):
        raise IndexError(f"layer_index {layer_index} out of range (have {len(layers)})")
    layer = layers[layer_index][1]
    phi = layer.phi
    if phi is None:
        raise RuntimeError(f"Layer {layer_index} has no learnable phi (use_learnable_phi=False)")

    device = device or next(phi.parameters()).device
    low, high = value_range
    grid = torch.linspace(low, high, grid_size, device=device)
    vi, vj = torch.meshgrid(grid, grid, indexing="ij")
    vi = vi.reshape(-1, 1).expand(-1, layer.in_channels)
    vj = vj.reshape(-1, 1).expand(-1, layer.in_channels)

    out = phi(vi, vj)  # (grid^2, C)
    out = out[:, channel].reshape(grid_size, grid_size)
    return out.detach().cpu()
