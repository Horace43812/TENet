"""Tool 2: Linear Centered Kernel Alignment.

Measures how much information from tensor X is linearly preserved in tensor Y.
Returns a scalar in [0, 1]; 1 means perfectly linearly predictable, 0 means
fully decorrelated.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited" (ICML 2019).
"""

from __future__ import annotations

import torch


def _flatten_to_matrix(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    elif x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.dim() != 2:
        raise ValueError(f"linear_cka expects 2/3/4-D tensor, got shape {tuple(x.shape)}")
    return x.float()


@torch.no_grad()
def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Linear CKA between activation tensors x and y.

    Both tensors must share the same first dim (sample count) after flattening.
    """
    a = _flatten_to_matrix(x)
    b = _flatten_to_matrix(y)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"linear_cka requires matched sample dim; got {a.shape[0]} vs {b.shape[0]}"
        )

    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    num = (b.T @ a).pow(2).sum()
    den = a.T.mm(a).norm() * b.T.mm(b).norm()
    return float((num / den.clamp_min(eps)).item())
