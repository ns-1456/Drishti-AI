"""
Phase 1: Fetch Sentinel-5P (NO2, CO, Aerosol Index) for Delhi NCR from Google Earth Engine.
GEE auth, NRTI L3 collections, date range, ROI clip, export to array.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Collections: NRTI L3; band names
S5P_COLLECTIONS = {
    "NO2": ("COPERNICUS/S5P/NRTI/L3_NO2", "tropospheric_NO2_column_number_density"),
    "CO": ("COPERNICUS/S5P/NRTI/L3_CO", "CO_column_number_density"),
    "AEROSOL_INDEX": ("COPERNICUS/S5P/NRTI/L3_AER_AI", "absorbing_aerosol_index"),
}


def get_gee_client() -> Any:
    """Initialize and return authenticated GEE client. Raise if not authenticated."""
    import ee
    try:
        ee.Initialize()
    except Exception as e:
        raise RuntimeError(
            "GEE not authenticated. Run: earthengine authenticate"
        ) from e
    return ee


def _roi_to_geometry(roi: dict[str, float]) -> Any:
    import ee
    return ee.Geometry.Rectangle([
        roi["min_lon"], roi["min_lat"],
        roi["max_lon"], roi["max_lat"],
    ])


def fetch_ncr_collection(
    product: str,
    start_date: date,
    end_date: date,
    roi: dict[str, float] | None = None,
) -> Any:
    """
    Fetch Sentinel-5P NRTI L3 collection for NCR ROI and date range.
    product: one of 'NO2', 'CO', 'AEROSOL_INDEX'
    Returns: ee.ImageCollection filtered by date and bounds.
    """
    import ee
    if product not in S5P_COLLECTIONS:
        raise ValueError(f"product must be one of {list(S5P_COLLECTIONS)}")
    col_id, band = S5P_COLLECTIONS[product]
    coll = ee.ImageCollection(col_id).select(band)
    start = start_date.isoformat()
    end = end_date.isoformat()
    coll = coll.filterDate(start, end)
    if roi:
        coll = coll.filterBounds(_roi_to_geometry(roi))
    return coll


def export_day_to_raster(
    product: str,
    day: date,
    roi: dict[str, float],
    out_path: Path | None = None,
    scale: int = 1000,
) -> np.ndarray:
    """
    Export a single day's product for ROI to in-memory array via sampleRegion/sampleRectangle.
    Returns (H, W) float array; may not be exactly 64x64 (use grid_align to resample).
    """
    import ee
    get_gee_client()
    if product not in S5P_COLLECTIONS:
        raise ValueError(f"product must be one of {list(S5P_COLLECTIONS)}")
    col_id, band = S5P_COLLECTIONS[product]
    geom = _roi_to_geometry(roi)
    day_start = day.isoformat()
    day_end = (day + timedelta(days=1)).isoformat()
    coll = ee.ImageCollection(col_id).select(band).filterDate(day_start, day_end).filterBounds(geom)
    image = coll.mean()
    image = image.reproject(crs="EPSG:4326", scale=scale)
    try:
        # sampleRectangle returns pixel values in a region (dict of band name -> 2D list)
        result = image.sampleRectangle(region=geom)
        arr = np.array(result.get(band).getInfo())
    except Exception:
        # Fallback: reduceRegion to a grid of points to get approximate array
        lon_len = int((roi["max_lon"] - roi["min_lon"]) * 111 * np.cos(np.radians(roi["min_lat"])) / (scale / 1000))
        lat_len = int((roi["max_lat"] - roi["min_lat"]) * 111 / (scale / 1000))
        lon_len, lat_len = max(8, min(lon_len, 256)), max(8, min(lat_len, 256))
        lons = np.linspace(roi["min_lon"], roi["max_lon"], lon_len)
        lats = np.linspace(roi["min_lat"], roi["max_lat"], lat_len)
        arr = np.zeros((len(lats), len(lons)), dtype=np.float32)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                pt = ee.Geometry.Point([lon, lat])
                val = image.reduceRegion(ee.Reducer.first(), pt, scale).get(band)
                try:
                    arr[i, j] = float(val.getInfo() or 0)
                except Exception:
                    arr[i, j] = np.nan
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if out_path:
        out_path = Path(out_path)
        if out_path.suffix == ".npy":
            np.save(out_path, arr)
        else:
            np.savetxt(out_path, arr)
    return arr.astype(np.float32)
