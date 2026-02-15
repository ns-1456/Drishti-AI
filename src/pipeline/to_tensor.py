"""
Phase 1: Normalize aligned grids and save as PyTorch .pt tensors.
Output: (T, 3, 64, 64) or one .pt per day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

# Sentinel-5P invalid/error values are often negative; product-specific ranges for clipping
PRODUCT_CLIP = {
    "NO2": (0.0, 1e-4),
    "CO": (0.0, 0.1),
    "AEROSOL_INDEX": (-2.0, 5.0),
}


def normalize_channel(
    data: np.ndarray,
    product: str,
    method: Literal["minmax", "robust", "zscore"] = "minmax",
) -> np.ndarray:
    """
    Normalize one channel. Mask invalid/negative (error) values; clip then scale.
    """
    out = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = PRODUCT_CLIP.get(product, (0.0, 1.0))
    out = np.clip(out, low, high)
    valid = np.isfinite(out) & (out >= low)
    if not np.any(valid):
        return out.astype(np.float32)
    if method == "minmax":
        vmin, vmax = np.min(out[valid]), np.max(out[valid])
        if vmax > vmin:
            out = (out - vmin) / (vmax - vmin)
    elif method == "robust":
        med, q75 = np.median(out[valid]), np.percentile(out[valid], 75)
        scale = max(q75 - med, 1e-9)
        out = (out - med) / scale
        out = np.clip(out, -3, 3)
    else:
        mu, sigma = np.mean(out[valid]), np.std(out[valid]) or 1e-9
        out = (out - mu) / sigma
    return out.astype(np.float32)


def stack_days_to_tensor(
    day_arrays: list[np.ndarray],
    channels_order: list[str] | None = None,
) -> np.ndarray:
    """
    Stack list of (C, 64, 64) arrays into (T, C, 64, 64), T = number of days.
    """
    if channels_order is None:
        channels_order = ["NO2", "CO", "AEROSOL_INDEX"]
    stacked = np.stack(day_arrays, axis=0)
    assert stacked.ndim == 4 and stacked.shape[2] == 64 and stacked.shape[3] == 64
    assert stacked.shape[1] == len(channels_order)
    return stacked


def save_tensor(tensor: np.ndarray | torch.Tensor, path: Path) -> None:
    """Save tensor as .pt (torch.save) or .npy."""
    path = Path(path)
    if path.suffix == ".pt":
        t = tensor if isinstance(tensor, torch.Tensor) else torch.from_numpy(np.asarray(tensor))
        torch.save(t, path)
    else:
        np.save(path, np.asarray(tensor))


def load_tensor(path: Path) -> np.ndarray | torch.Tensor:
    """Load .pt or .npy tensor from path."""
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, weights_only=True)
    return np.load(path)
