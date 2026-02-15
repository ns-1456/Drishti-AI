#!/usr/bin/env python3
"""
Satellite baseline: download last 30 days of NO2, CO, Aerosol Index for Delhi NCR from GEE.
Run after: earthengine authenticate
Saves (30, 3, 64, 64) to data/satellite/ncr_30d.pt for Vision Mamba training.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.gee_client import export_day_to_raster, get_gee_client
from src.pipeline.grid_align import align_day, load_roi_config
from src.pipeline.to_tensor import normalize_channel, save_tensor, stack_days_to_tensor

CHANNELS_ORDER = ["NO2", "CO", "AEROSOL_INDEX"]


def main():
    """Fetch last 30 days; align to 64x64; save .pt."""
    get_gee_client()
    cfg = load_roi_config()
    roi = cfg["roi"]
    target_shape = (cfg["grid"]["height"], cfg["grid"]["width"])
    products = cfg.get("products", CHANNELS_ORDER)
    end = date.today()
    start = end - timedelta(days=30)

    day_arrays = []
    for d in range((end - start).days):
        day = start + timedelta(days=d)
        channels = []
        for product in products:
            arr = export_day_to_raster(product, day, roi)
            aligned = align_day(raster_array=arr, roi=roi, target_shape=target_shape)
            aligned = aligned[0] if aligned.ndim == 3 else aligned
            norm = normalize_channel(aligned, product, method="minmax")
            channels.append(norm)
        stack_day = np.stack(channels, axis=0)
        day_arrays.append(stack_day)

    tensor = stack_days_to_tensor(day_arrays, channels_order=products)
    out_dir = ROOT / "data" / "satellite"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ncr_30d.pt"
    save_tensor(tensor, out_path)
    print(f"Saved {tensor.shape} to {out_path}")


if __name__ == "__main__":
    main()
