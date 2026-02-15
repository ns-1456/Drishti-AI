#!/usr/bin/env python3
"""
Offline ETL pipeline: batch historical data for training.
Fetches satellite (GEE), aligns to grid, normalizes, and writes training dataset.
Use in data-engineering project: run on a schedule (e.g. weekly) to refresh "all data until now".
Output: data/satellite/ and optionally parquet/CSV for social/CPCB when added.
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


def run_offline_satellite(
    start_date: date,
    end_date: date,
    out_dir: Path,
    config_path: Path | None = None,
) -> Path:
    """Build offline satellite tensor for date range. Returns path to .pt file."""
    get_gee_client()
    cfg = load_roi_config(config_path)
    roi = cfg["roi"]
    target_shape = (cfg["grid"]["height"], cfg["grid"]["width"])
    products = cfg.get("products", CHANNELS_ORDER)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_arrays = []
    days = (end_date - start_date).days
    for d in range(days):
        day = start_date + timedelta(days=d)
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
    out_path = out_dir / f"ncr_offline_{start_date.isoformat()}_{end_date.isoformat()}.pt"
    save_tensor(tensor, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Offline ETL: batch historical satellite for NCR")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="Number of days if start/end not set")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    end = date.today() if not args.end else date.fromisoformat(args.end)
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        start = end - timedelta(days=args.days)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "satellite"

    path = run_offline_satellite(start, end, out_dir)
    print(f"Offline pipeline done: {path}")


if __name__ == "__main__":
    main()
