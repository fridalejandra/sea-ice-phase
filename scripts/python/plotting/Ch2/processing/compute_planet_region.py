#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_planet_region.py

Identifies the best region and time window to request Planet imagery
for qualitative validation of phase transition structure.

Strategy:
  1. Convert SMMR polar stereographic grid to lat/lon
  2. Mask to Weddell Sea and East Antarctica coast sectors
  3. Load method-spread ambiguity (std across all static thr/k combos)
  4. Find grid cells with largest AND most consistent ambiguity across years
  5. Identify the year where ambiguity peaks at that location
  6. Extract the static vs dynamic phase dates for that cell/year
  7. Output: lat/lon bounding box + date range for Planet request

Usage:
  python compute_planet_region.py
  python compute_planet_region.py --phase MS   # default FS
  python compute_planet_region.py --sector EA  # default Weddell
"""

import argparse
import numpy as np
import xarray as xr
from pyproj import Transformer

# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT    = "/user/geog/falejandraperez/sea-ice-phase/data"
METRICS_DIR  = f"{DATA_ROOT}/transition_metrics/SMMR"
PHASE_DIR    = f"{DATA_ROOT}/SMMR_phase"
MERGED_FILE  = f"{DATA_ROOT}/merged/SMMR_merged_19781101_20251231.nc"

# Sector lat/lon bounds (approximate)
SECTORS = {
    "Weddell": {
        "lon_min": -60,  "lon_max":  20,
        "lat_min": -80,  "lat_max": -60,
        "description": "Weddell Sea coast",
    },
    "EA": {
        "lon_min":  60,  "lon_max": 150,
        "lat_min": -70,  "lat_max": -60,
        "description": "East Antarctica coast",
    },
    "Ross": {
        "lon_min": 160,  "lon_max": -140,  # wraps dateline
        "lat_min": -80,  "lat_max": -65,
        "description": "Ross Sea",
    },
}

# Planet bounding box padding (degrees)
BOX_PAD = 1.5

# Date window padding around the transition (days)
DATE_PAD = 14


# =============================================================================
# GRID → LAT/LON
# =============================================================================

def get_latlon_grid(merged_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert NSIDC 25km Southern Hemisphere polar stereographic grid to lat/lon.
    Uses pyproj with the standard NSIDC EPSG:3976 projection.
    """
    ds = xr.open_dataset(merged_file)
    x  = ds.x.values   # (nx,) in meters
    y  = ds.y.values   # (ny,) in meters
    ds.close()

    # meshgrid
    xx, yy = np.meshgrid(x, y)   # both (ny, nx)

    # NSIDC Southern Hemisphere 25km: EPSG:3976
    transformer = Transformer.from_crs("EPSG:3976", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)

    return lat, lon   # both (ny, nx)


# =============================================================================
# SECTOR MASK
# =============================================================================

def sector_mask(lat: np.ndarray, lon: np.ndarray,
                sector: str) -> np.ndarray:
    """Return boolean mask for a named sector."""
    s = SECTORS[sector]
    lat_ok = (lat >= s["lat_min"]) & (lat <= s["lat_max"])

    if sector == "Ross":  # wraps dateline
        lon_ok = (lon >= s["lon_min"]) | (lon <= s["lon_max"])
    else:
        lon_ok = (lon >= s["lon_min"]) & (lon <= s["lon_max"])

    return lat_ok & lon_ok


# =============================================================================
# FIND BEST REGION
# =============================================================================

def find_best_region(phase: str, sector: str) -> dict:
    """
    Find the grid cell and year with the largest, most consistent
    method-spread ambiguity within the sector.

    Returns a dict with:
      lat, lon        -- center of best cell
      bbox            -- (lat_min, lat_max, lon_min, lon_max) for Planet request
      best_year       -- year of peak ambiguity
      static_date     -- static thr15 k5 phase date (DOY) for that year
      dynamic_date    -- dynamic phase date (DOY) for that year
      date_range      -- (start_date, end_date) string for Planet request
      ambiguity_val   -- ambiguity value at best cell
    """
    print(f"\nComputing lat/lon grid...")
    lat, lon = get_latlon_grid(MERGED_FILE)

    print(f"Masking to {sector} sector...")
    mask = sector_mask(lat, lon, sector)
    print(f"  {mask.sum()} grid cells in sector")

    print(f"Loading method-spread ambiguity for {phase}...")
    amb_ds   = xr.open_dataset(f"{METRICS_DIR}/method_spread_{phase}.nc")
    amb      = amb_ds["method_spread"].values   # (year, y, x)
    years    = amb_ds["year"].values
    amb_ds.close()

    # mean ambiguity across years — find persistently ambiguous cells
    amb_mean = np.nanmean(amb, axis=0)   # (y, x)

    # apply sector mask
    amb_mean_masked = np.where(mask, amb_mean, np.nan)

    # find best cell
    best_flat = np.nanargmax(amb_mean_masked)
    best_j, best_i = np.unravel_index(best_flat, amb_mean.shape)

    best_lat = float(lat[best_j, best_i])
    best_lon = float(lon[best_j, best_i])
    best_amb = float(amb_mean_masked[best_j, best_i])

    print(f"\nBest cell: lat={best_lat:.2f} lon={best_lon:.2f}")
    print(f"  Mean ambiguity: {best_amb:.1f} days")

    # find year of peak ambiguity at that cell
    amb_at_cell = amb[:, best_j, best_i]
    best_year_idx = np.nanargmax(amb_at_cell)
    best_year = int(years[best_year_idx])
    print(f"  Peak year: {best_year} (ambiguity={amb_at_cell[best_year_idx]:.1f} days)")

    # load static and dynamic dates for that year at that cell
    static_path  = (f"{PHASE_DIR}/static/thr15_k5/{phase}/{phase}_{best_year}.nc")
    dynamic_path = (f"{PHASE_DIR}/dynamic/k5_q70/{phase}/{phase}_{best_year}.nc")

    static_doy  = np.nan
    dynamic_doy = np.nan

    try:
        ds = xr.open_dataset(static_path)
        static_doy = float(ds[phase].values[best_j, best_i])
        ds.close()
    except Exception as e:
        print(f"  Warning: could not load static date: {e}")

    try:
        ds = xr.open_dataset(dynamic_path)
        dynamic_doy = float(ds[phase].values[best_j, best_i])
        ds.close()
    except Exception as e:
        print(f"  Warning: could not load dynamic date: {e}")

    print(f"  Static  {phase} DOY: {static_doy:.0f}")
    print(f"  Dynamic {phase} DOY: {dynamic_doy:.0f}")
    print(f"  Difference: {abs(static_doy - dynamic_doy):.0f} days")

    # build date range centered on midpoint of static/dynamic dates
    if np.isfinite(static_doy) and np.isfinite(dynamic_doy):
        mid_doy = int((static_doy + dynamic_doy) / 2)
    elif np.isfinite(static_doy):
        mid_doy = int(static_doy)
    else:
        mid_doy = int(dynamic_doy)

    # convert DOY to date string
    import datetime
    # handle MS which can cross year boundary
    ref_year = best_year if phase == "FS" else (
        best_year if mid_doy <= 365 else best_year + 1)
    try:
        mid_date  = datetime.date(ref_year, 1, 1) + datetime.timedelta(days=mid_doy - 1)
        start_date = mid_date - datetime.timedelta(days=DATE_PAD)
        end_date   = mid_date + datetime.timedelta(days=DATE_PAD)
        date_range = (start_date.isoformat(), end_date.isoformat())
    except Exception:
        date_range = ("unknown", "unknown")

    # bounding box
    bbox = (
        round(best_lat - BOX_PAD, 2),
        round(best_lat + BOX_PAD, 2),
        round(best_lon - BOX_PAD, 2),
        round(best_lon + BOX_PAD, 2),
    )

    return {
        "lat": best_lat,
        "lon": best_lon,
        "bbox": bbox,
        "best_year": best_year,
        "static_doy": static_doy,
        "dynamic_doy": dynamic_doy,
        "date_range": date_range,
        "ambiguity_days": best_amb,
        "sector": sector,
        "phase": phase,
    }


# =============================================================================
# REPORT
# =============================================================================

def print_report(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Planet Imagery Request")
    print(f"{'='*60}")
    print(f"  Phase:       {result['phase']}")
    print(f"  Sector:      {result['sector']} ({SECTORS[result['sector']]['description']})")
    print(f"  Center:      lat={result['lat']:.2f}°  lon={result['lon']:.2f}°")
    print(f"  Bounding box:")
    print(f"    lat: {result['bbox'][0]}° to {result['bbox'][1]}°")
    print(f"    lon: {result['bbox'][2]}° to {result['bbox'][3]}°")
    print(f"  Target year: {result['best_year']}")
    print(f"  Static  DOY: {result['static_doy']:.0f}")
    print(f"  Dynamic DOY: {result['dynamic_doy']:.0f}")
    print(f"  Disagreement: {abs(result['static_doy'] - result['dynamic_doy']):.0f} days")
    print(f"  Date range:  {result['date_range'][0]}  →  {result['date_range'][1]}")
    print(f"  Mean ambiguity: {result['ambiguity_days']:.1f} days")
    print(f"{'='*60}")
    print(f"\nSend to friend:")
    print(f"  bbox:  {result['bbox']}")
    print(f"  dates: {result['date_range'][0]} to {result['date_range'][1]}")
    print(f"  year:  {result['best_year']}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Find best region and dates for Planet imagery request.")
    p.add_argument("--phase",  default="FS", choices=["FS", "MS"])
    p.add_argument("--sector", default="Weddell",
                   choices=["Weddell", "EA", "Ross"])
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    result = find_best_region(args.phase, args.sector)
    print_report(result)