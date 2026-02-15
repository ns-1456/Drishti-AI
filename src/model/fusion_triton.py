"""
Phase 3: Fusion layer — satellite features + social point features.
PyTorch cross-attention with sparse mask (satellite cells x nearby social posts).
Optional: Triton block-sparse kernel for <10 ms latency.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_sparse_attention_mask(
    satellite_grid_shape: tuple[int, int],
    social_lat_lon: Any,
    roi: dict[str, float],
    grid_height: int = 64,
    grid_width: int = 64,
    radius_cells: int = 2,
) -> torch.Tensor:
    """
    Build boolean mask: (H*W, num_social_posts).
    True only for (cell, post) pairs where post is within radius_cells of cell.
    social_lat_lon: (N, 2) or (B, N, 2) in [lat, lon] or [lon, lat]; we assume [lon, lat].
    """
    min_lon = roi["min_lon"]
    min_lat = roi["min_lat"]
    max_lon = roi["max_lon"]
    max_lat = roi["max_lat"]
    if isinstance(social_lat_lon, torch.Tensor):
        plon = social_lat_lon[..., 0]
        plat = social_lat_lon[..., 1]
    else:
        import numpy as np
        arr = np.asarray(social_lat_lon)
        plon = arr[..., 0]
        plat = arr[..., 1]
    # Map to cell indices [0, grid_height-1], [0, grid_width-1]
    cell_h = (plat - min_lat) / (max_lat - min_lat + 1e-9) * (grid_height - 1)
    cell_w = (plon - min_lon) / (max_lon - min_lon + 1e-9) * (grid_width - 1)
    cell_h = torch.clamp(torch.as_tensor(cell_h, dtype=torch.float32), 0, grid_height - 1)
    cell_w = torch.clamp(torch.as_tensor(cell_w, dtype=torch.float32), 0, grid_width - 1)
    if cell_h.dim() == 1:
        cell_h = cell_h.unsqueeze(0)
        cell_w = cell_w.unsqueeze(0)
    B, N = cell_h.shape
    num_cells = grid_height * grid_width
    mask = torch.zeros(B, num_cells, N, dtype=torch.bool, device=cell_h.device)
    for b in range(B):
        for j in range(N):
            ch, cw = int(cell_h[b, j].item()), int(cell_w[b, j].item())
            for di in range(-radius_cells, radius_cells + 1):
                for dj in range(-radius_cells, radius_cells + 1):
                    nh, nw = ch + di, cw + dj
                    if 0 <= nh < grid_height and 0 <= nw < grid_width:
                        idx = nh * grid_width + nw
                        mask[b, idx, j] = True
    return mask


class SparseFusionLayer(nn.Module):
    """
    Cross-attention: satellite (queries) attend to social (keys/values) with sparse mask.
    """

    def __init__(
        self,
        satellite_dim: int,
        social_dim: int,
        num_heads: int = 8,
        use_triton: bool = False,
    ) -> None:
        super().__init__()
        self.use_triton = use_triton
        self.satellite_dim = satellite_dim
        self.social_dim = social_dim
        self.num_heads = num_heads
        assert satellite_dim % num_heads == 0 and social_dim % num_heads == 0
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=satellite_dim,
            num_heads=num_heads,
            kdim=social_dim,
            vdim=social_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(satellite_dim)

    def forward(
        self,
        satellite_feat: torch.Tensor,
        social_feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        satellite_feat: (B, H*W, D_sat). social_feat: (B, N_posts, D_soc).
        mask: (B, H*W, N_posts) — True where attend. Return: (B, H*W, D_sat).
        """
        # attn_mask: (Lq, Lk) with -inf where not to attend
        if mask.dim() == 3:
            attn_mask = torch.where(mask, 0.0, -1e9).float()
            attn_mask = attn_mask[0]
        else:
            attn_mask = torch.where(mask, 0.0, -1e9).float()
        attn_out, _ = self.cross_attn(
            satellite_feat,
            social_feat,
            social_feat,
            attn_mask=attn_mask,
        )
        return self.norm(satellite_feat + attn_out)


# Optional: Triton kernel for block-sparse attention (portfolio piece)
# def triton_sparse_attention(q, k, v, mask): ...
# Target: <10 ms for typical NCR grid + social post count.
