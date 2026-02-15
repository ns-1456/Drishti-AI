"""
Phase 1: Align variable-resolution satellite data to fixed 1 km x 1 km grid (64x64) over NCR.
Bicubic interpolation from ~3.5 km native resolution to 64x64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import ndimage


def load_roi_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load ROI and grid settings from config/roi.yaml."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "roi.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "roi": cfg["roi"],
        "grid": cfg["grid"],
        "products": cfg.get("products", ["NO2", "CO", "AEROSOL_INDEX"]),
    }


def raster_to_grid(
    data: np.ndarray,
    src_bounds: tuple[float, float, float, float] | None = None,
    target_shape: tuple[int, int] = (64, 64),
    roi: dict[str, float] | None = None,
    method: str = "bicubic",
) -> np.ndarray:
    """
    Resample raster to fixed grid (e.g. 64x64) over ROI using bicubic interpolation.
    data: 2D array (H, W) for one product; or 3D (C, H, W) for multiple products.
    Returns: array of shape (C, 64, 64) or (64, 64).
    """
    order = 3 if method == "bicubic" else 1
    if data.ndim == 2:
        h, w = data.shape
        zoom = (target_shape[0] / h, target_shape[1] / w)
        out = ndimage.zoom(data.astype(np.float64), zoom, order=order)
        return out.astype(np.float32)
    else:
        c, h, w = data.shape
        zoom = (1, target_shape[0] / h, target_shape[1] / w)
        out = ndimage.zoom(data.astype(np.float64), zoom, order=order)
        return out.astype(np.float32)


def align_day(
    raster_path: Path | None = None,
    raster_array: np.ndarray | None = None,
    roi: dict[str, float] | None = None,
    target_shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Align one day's raster (single or multi-band) to target grid.
    Returns: (C, 64, 64) float array.
    """
    if raster_array is not None:
        data = raster_array
    elif raster_path is not None:
        try:
            import rasterio
            with rasterio.open(raster_path) as src:
                data = src.read()
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[0]
        except Exception:
            data = np.load(raster_path) if raster_path.suffix == ".npy" else np.loadtxt(raster_path)
    else:
        raise ValueError("Provide raster_path or raster_array")
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    out = raster_to_grid(data, target_shape=target_shape, method="bicubic")
    return out
