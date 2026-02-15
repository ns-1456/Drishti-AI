"""
Unified fusion model: Vim + SocialPointEncoder + SparseFusion + regression head.
Output: final PM2.5 prediction map (B, H, W).
Optional modalities: meteo, fire, past_aqi, land_use (extra grid channels or None).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.model.fusion_triton import build_sparse_attention_mask, SparseFusionLayer
from src.model.point_net import SocialPointEncoder
from src.model.vim import VimBaseEncoder


class FusionModel(nn.Module):
    """
    Satellite (Vim) + social (PointNet) + sparse fusion + regression head.
    forward() returns (B, H, W) PM2.5 prediction map.
    """

    def __init__(
        self,
        # Satellite
        in_channels: int = 3,
        satellite_dim: int = 768,
        patch_size: int = 4,
        vim_depth: int = 4,
        grid_height: int = 64,
        grid_width: int = 64,
        # Social
        clip_dim: int = 512,
        social_dim: int = 768,
        # Fusion
        num_heads: int = 8,
        # Optional extra grid channels (meteo, fire, past_aqi, land_use)
        meteo_channels: int = 0,
        fire_channels: int = 0,
        past_aqi_channels: int = 0,
        land_use_channels: int = 0,
    ) -> None:
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_cells = grid_height * grid_width
        total_grid_channels = in_channels + meteo_channels + fire_channels + past_aqi_channels + land_use_channels

        self.vim = VimBaseEncoder(
            in_channels=total_grid_channels,
            dim=satellite_dim,
            patch_size=patch_size,
            depth=vim_depth,
        )
        self.social_encoder = SocialPointEncoder(clip_dim=clip_dim, out_dim=social_dim)
        self.fusion = SparseFusionLayer(
            satellite_dim=satellite_dim,
            social_dim=social_dim,
            num_heads=num_heads,
        )
        self.head = nn.Sequential(
            nn.Linear(satellite_dim, satellite_dim // 2),
            nn.GELU(),
            nn.Linear(satellite_dim // 2, 1),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        social_points: torch.Tensor,
        social_lat_lon: torch.Tensor,
        roi: dict[str, float],
        social_mask: torch.Tensor | None = None,
        meteo_grid: torch.Tensor | None = None,
        fire_grid: torch.Tensor | None = None,
        past_aqi_grid: torch.Tensor | None = None,
        land_use_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        satellite: (B, 3, H, W). social_points: (B, N, clip_dim). social_lat_lon: (B, N, 2).
        roi: {min_lon, min_lat, max_lon, max_lat}.
        Returns: (B, H, W) PM2.5 prediction map.
        """
        B, _, H, W = satellite.shape
        grids = [satellite]
        if meteo_grid is not None:
            grids.append(meteo_grid)
        if fire_grid is not None:
            grids.append(fire_grid)
        if past_aqi_grid is not None:
            grids.append(past_aqi_grid)
        if land_use_grid is not None:
            grids.append(land_use_grid)
        grid_input = torch.cat(grids, dim=1)

        satellite_feat = self.vim(grid_input)
        social_feat = self.social_encoder(social_points, social_mask)

        mask = build_sparse_attention_mask(
            (self.grid_height, self.grid_width),
            social_lat_lon,
            roi,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        if mask.device != satellite_feat.device:
            mask = mask.to(satellite_feat.device)

        fused = self.fusion(satellite_feat, social_feat, mask)
        out = self.head(fused)
        out = out.squeeze(-1).view(B, self.grid_height, self.grid_width)
        return out
