"""Interchangeable token-pruning modules for the ViT mid-network slot."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class TokenPruner(nn.Module, ABC):
    """Contract for a module receiving tokens including the CLS token at index 0."""

    @abstractmethod
    def forward(self, tokens: Tensor) -> Tensor:
        """Return tokens with the CLS token retained as the first token."""


class IdentityTokenPruner(TokenPruner):
    """No-pruning reference method for controlled baseline experiments."""

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens


class MagnitudeTokenPruner(TokenPruner):
    """Keep patch tokens with the largest L2 feature norm.

    This parameter-free baseline actually removes tokens, so all subsequent
    Transformer blocks perform less attention and MLP computation. Selection is
    per sample; selected token indices are sorted to retain raster order.
    """

    def __init__(self, keep_ratio: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = float(keep_ratio)
        self.last_keep_indices: Tensor | None = None

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.size(1) < 2:
            raise ValueError("Expected (batch, cls + patch_tokens, channels) with at least one patch token")
        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        keep_count = max(1, int(round(patch_tokens.size(1) * self.keep_ratio)))
        scores = patch_tokens.square().mean(dim=-1)
        keep_indices = scores.topk(keep_count, dim=1, sorted=False).indices.sort(dim=1).values
        self.last_keep_indices = keep_indices.detach()
        gather_indices = keep_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
        return torch.cat((cls_token, patch_tokens.gather(dim=1, index=gather_indices)), dim=1)


class LinearTokenPruner(TokenPruner):
    """Learn a scalar token-importance score with a single linear layer.

    Top-K index selection is non-differentiable. During training, selected
    tokens use a straight-through sigmoid gate: the forward pass remains a
    hard Top-K prune, while the scorer receives a surrogate gradient. During
    evaluation the gate is exactly one for retained tokens.
    """

    def __init__(self, embed_dim: int, keep_ratio: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = float(keep_ratio)
        self.score = nn.Linear(embed_dim, 1)
        self.last_scores: Tensor | None = None
        self.last_keep_indices: Tensor | None = None

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.size(1) < 2:
            raise ValueError("Expected (batch, cls + patch_tokens, channels) with at least one patch token")
        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        scores = self.score(patch_tokens).squeeze(-1)
        keep_count = max(1, int(round(patch_tokens.size(1) * self.keep_ratio)))
        keep_indices = scores.topk(keep_count, dim=1, sorted=False).indices.sort(dim=1).values
        self.last_scores = scores.detach()
        self.last_keep_indices = keep_indices.detach()
        gather_indices = keep_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
        kept_tokens = patch_tokens.gather(dim=1, index=gather_indices)
        if self.training:
            hard_mask = torch.zeros_like(scores).scatter_(1, keep_indices, 1.0)
            soft_mask = scores.sigmoid()
            straight_through_mask = hard_mask + soft_mask - soft_mask.detach()
            kept_gate = straight_through_mask.gather(dim=1, index=keep_indices).unsqueeze(-1)
            kept_tokens = kept_tokens * kept_gate
        return torch.cat((cls_token, kept_tokens), dim=1)


class TENetLocalTokenPruner(TokenPruner):
    """Parameter-free 2x2 local graph scorer derived from TENet.

    A 2x2 token window forms four vertex nodes. The six absolute-difference
    features are graph edge nodes, producing the same 10-node incidence graph
    as TENet. For vertex ``i`` its score is its local contribution to the
    Laplacian (Dirichlet) energy, summed over adjacent edge nodes:
    ``sum_j A[i,j] * mean_c((z_i[c] - z_j[c])**2)``.
    """

    def __init__(self, grid_size: int, keep_ratio: float = 0.5) -> None:
        super().__init__()
        if grid_size <= 0 or grid_size % 2:
            raise ValueError("TENet local pruning requires a positive, even token grid size")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.grid_size = int(grid_size)
        self.keep_ratio = float(keep_ratio)
        self.register_buffer("laplacian", self._build_laplacian())
        self.last_scores: Tensor | None = None
        self.last_keep_indices: Tensor | None = None

    @staticmethod
    def _build_laplacian() -> Tensor:
        adjacency = torch.zeros(10, 10)
        # Four vertex nodes (0..3), six edge-feature nodes (4..9).
        for source, target in [(0, 4), (1, 4), (0, 5), (2, 5), (0, 6), (3, 6),
                               (1, 7), (2, 7), (1, 8), (3, 8), (2, 9), (3, 9)]:
            adjacency[source, target] = adjacency[target, source] = 1.0
        return torch.diag(adjacency.sum(dim=1)) - adjacency

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.size(1) != self.grid_size ** 2 + 1:
            raise ValueError(
                f"Expected CLS + {self.grid_size ** 2} patch tokens for a {self.grid_size}x{self.grid_size} grid"
            )
        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        batch, _, channels = patch_tokens.shape
        grid = patch_tokens.reshape(batch, self.grid_size, self.grid_size, channels)
        half = self.grid_size // 2
        vertices = grid.reshape(batch, half, 2, half, 2, channels)
        vertices = vertices.permute(0, 1, 3, 2, 4, 5).reshape(batch, half, half, 4, channels)
        v1, v2, v3, v4 = vertices.unbind(dim=3)
        edge_features = torch.stack(
            [torch.abs(v1 - v2), torch.abs(v1 - v3), torch.abs(v1 - v4),
             torch.abs(v2 - v3), torch.abs(v2 - v4), torch.abs(v3 - v4)], dim=3
        )
        z = torch.cat((vertices, edge_features), dim=3)

        # ``L < 0`` gives exactly the 12 vertex-to-edge adjacency links.
        link_pairs = (self.laplacian < 0).nonzero(as_tuple=False)
        vertex_scores = torch.zeros(batch, half, half, 4, device=tokens.device, dtype=tokens.dtype)
        for source, target in link_pairs.tolist():
            if source < 4:
                energy = (z[:, :, :, source] - z[:, :, :, target]).square().mean(dim=-1)
                vertex_scores[:, :, :, source] += energy

        scores = vertex_scores.reshape(batch, half, half, 2, 2)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch, self.grid_size ** 2)
        keep_count = max(1, int(round(scores.size(1) * self.keep_ratio)))
        keep_indices = scores.topk(keep_count, dim=1, sorted=False).indices.sort(dim=1).values
        self.last_scores = scores.detach()
        self.last_keep_indices = keep_indices.detach()
        gather_indices = keep_indices.unsqueeze(-1).expand(-1, -1, channels)
        kept_tokens = patch_tokens.gather(dim=1, index=gather_indices)
        return torch.cat((cls_token, kept_tokens), dim=1)


def build_token_pruner(
    name: str,
    keep_ratio: float = 0.5,
    embed_dim: int | None = None,
    grid_size: int | None = None,
) -> TokenPruner:
    """Build a registered pruner; add future methods here without editing ViT."""
    if name == "identity":
        return IdentityTokenPruner()
    if name == "magnitude":
        return MagnitudeTokenPruner(keep_ratio)
    if name == "linear":
        if embed_dim is None:
            raise ValueError("embed_dim is required for the linear token pruner")
        return LinearTokenPruner(embed_dim, keep_ratio)
    if name == "tenet_local":
        if grid_size is None:
            raise ValueError("grid_size is required for the TENet local token pruner")
        return TENetLocalTokenPruner(grid_size, keep_ratio)
    raise ValueError(f"Unknown token pruner: {name}. Available: identity, magnitude, linear, tenet_local")
