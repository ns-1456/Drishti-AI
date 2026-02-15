#!/usr/bin/env python3
"""
Online ETL pipeline: fetch latest N days for inference (and optional CL update).
Use in data-engineering project: run daily or every N minutes to get fresh inputs.
Output: single-day or small-window tensor for inference; same format as offline for training.
"""

from __future__ import annotations

import argparse
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


def run_online_satellite(
    for_date: date | None = None,
    last_n_days: int = 1,
    out_dir: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    """Build online satellite tensor for latest day(s). Returns path to .pt file."""
    get_gee_client()
    cfg = load_roi_config(config_path)
    roi = cfg["roi"]
    target_shape = (cfg["grid"]["height"], cfg["grid"]["width"])
    products = cfg.get("products", CHANNELS_ORDER)
    out_dir = Path(out_dir) if out_dir else ROOT / "data" / "satellite" / "online"
    out_dir.mkdir(parents=True, exist_ok=True)

    end = for_date or date.today()
    start = end - timedelta(days=last_n_days - 1)
    day_arrays = []
    for d in range(last_n_days):
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
    out_path = out_dir / f"ncr_online_{end.isoformat()}.pt"
    save_tensor(tensor, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Online ETL: latest satellite for inference")
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD (default today)")
    parser.add_argument("--last-n-days", type=int, default=1, help="Window size in days")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    for_date = date.fromisoformat(args.date) if args.date else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    path = run_online_satellite(for_date=for_date, last_n_days=args.last_n_days, out_dir=out_dir)
    print(f"Online pipeline done: {path}")


if __name__ == "__main__":
    main()
