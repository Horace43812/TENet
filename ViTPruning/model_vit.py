"""A small, standard Vision Transformer baseline.

The module names intentionally follow the common ViT convention:
``patch_embed.proj``, ``blocks.<n>.attn.qkv``, ``attn.proj``,
``mlp.fc1`` and ``mlp.fc2``.  This makes the model a convenient target for
both unstructured and structured pruning experiments.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
from torch import Tensor, nn

from pruning import TokenPruner, build_token_pruner


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and linearly embed them."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) input, got shape {tuple(x.shape)}")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} input, got {x.shape[-2:]}"
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block used by the original ViT."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """Minimal ViT classifier with fixed learned positional embeddings.

    The token-pruner slot runs after ``prune_at_block`` encoder blocks. It is
    the sole integration point for token-pruning methods, keeping the ViT
    backbone independent from individual pruning algorithms.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pruner_name: str = "magnitude",
        prune_keep_ratio: float = 0.5,
        prune_at_block: int | None = None,
        token_pruner: TokenPruner | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.prune_at_block = depth // 2 if prune_at_block is None else int(prune_at_block)
        if not 1 <= self.prune_at_block <= depth:
            raise ValueError("prune_at_block must be between 1 and depth (inclusive)")
        self.token_pruner = token_pruner if token_pruner is not None else build_token_pruner(
            pruner_name, prune_keep_ratio, embed_dim, self.patch_embed.grid_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls_token, x), dim=1) + self.pos_embed)
        for block_index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if block_index == self.prune_at_block:
                x = self.token_pruner(x)
        return self.norm(x)[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))

    def iter_prunable_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Yield Conv2d/Linear layers, excluding LayerNorm and parameters tokens."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                yield name, module


def vit_tiny_cifar10(**kwargs: object) -> VisionTransformer:
    """Standard compact baseline: 4x4 patches, 192 dim, 12 blocks, 3 heads."""
    defaults = dict(image_size=32, patch_size=4, in_channels=3, num_classes=10,
                    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1,
                    pruner_name="magnitude", prune_keep_ratio=0.5, prune_at_block=6)
    defaults.update(kwargs)
    return VisionTransformer(**defaults)
