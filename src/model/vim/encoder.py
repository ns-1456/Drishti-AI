"""
Phase 3: Vision Mamba (Vim-Base) backbone for satellite stream.
Minimal implementation: patchify + linear layers (placeholder for full Mamba blocks).
Input: (B, C, 64, 64). Output: (B, N, D) for fusion.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class VimBaseEncoder(nn.Module):
    """Minimal Vim-style encoder: patchify then project to dim. (B, 3, 64, 64) -> (B, N, D)."""

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 768,
        patch_size: int = 4,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (64 // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4, batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 64, 64). Return (B, N, D)."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
