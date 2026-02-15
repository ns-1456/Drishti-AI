"""
Phase 3: Point-Net style encoder for scattered social CLIP embeddings.
MLP per point + max pool over points.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SocialPointEncoder(nn.Module):
    """
    Encode variable-length set of CLIP embeddings: MLP per point + max pool.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        hidden_dim: int = 256,
        out_dim: int = 768,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        points: (B, N, clip_dim). mask: (B, N) 1=valid, 0=pad.
        Return: (B, N, out_dim) per-point features for fusion.
        """
        out = self.mlp(points)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out
