"""Tool 1: Effective Rank via singular-value entropy.

For a feature tensor X reshaped to (N, C), compute the normalised singular-value
distribution and report exp(H(p)) where H is Shannon entropy. The result lives
in [1, C]; values << C indicate the layer's effective bandwidth is collapsing.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def effective_rank(x: torch.Tensor, center: bool = True, eps: float = 1e-12) -> float:
    """Return the effective rank of a (..., C) feature tensor.

    Args:
        x: tensor of shape (B, C, H, W), (B, N, C) or (N, C). The last
           non-singleton dimension is treated as the channel dim if x has
           exactly 2 dims; otherwise channels are taken from dim=1 for 4D
           tensors and from dim=-1 for 3D tensors.
        center: whether to subtract the mean per channel before SVD.
        eps: numerical floor.
    """
    if x.dim() == 4:
        # (B, C, H, W) -> (B*H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    elif x.dim() == 3:
        # (B, N, C) -> (B*N, C)
        x = x.reshape(-1, x.shape[-1])
    elif x.dim() != 2:
        raise ValueError(f"effective_rank expects 2/3/4-D tensor, got shape {tuple(x.shape)}")

    x = x.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)

    if x.shape[0] < 2:
        return 1.0

    s = torch.linalg.svdvals(x)
    total = s.sum()
    if total <= eps:
        return 1.0
    p = s / total
    entropy = -(p * (p + eps).log()).sum()
    return float(entropy.exp().item())


@torch.no_grad()
def effective_rank_per_channel(x: torch.Tensor) -> dict:
    """Convenience wrapper returning rank along with raw shape info."""
    er = effective_rank(x)
    if x.dim() == 4:
        c = x.shape[1]
    elif x.dim() == 3:
        c = x.shape[-1]
    else:
        c = x.shape[-1]
    return {"effective_rank": er, "num_channels": int(c), "ratio": er / max(int(c), 1)}
