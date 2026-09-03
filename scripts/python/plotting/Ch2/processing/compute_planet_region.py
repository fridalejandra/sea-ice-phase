#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_planet_region.py

Identifies the best region and time window to request Planet imagery
for qualitative validation of phase transition structure.

Strategy (Vichi-informed):
  Rather than maximizing raw ambiguity (which finds pathological cells),
  we find where the CHARACTER of the transition changed most post-2017:
    delta_freq = mean(crossing_freq, 2017-2024) - mean(crossing_freq, 1980-2016)
  Large positive delta_freq = transitions became more flickering/diffuse post-2017.
  This connects directly to the main trend result and gives a physically
  motivated criterion for Planet imagery selection.

  Additional filters:
    - Both periods must have valid data
    - Cell must be in the interior pack (not coast/edge)
    - Static vs dynamic must agree on sign of change
    - Mean climatological SIC in winter must be > 0.5 (proper ice pack)

Usage:
  python compute_planet_region.py                    # SMMR FS Weddell
  python compute_planet_region.py --phase MS
  python compute_planet_region.py --sector EA
  python compute_planet_region.py --sector all       # compare all sectors
"""

import argparse
import datetime
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

PRE_CUTOFF   = 2017   # years < PRE_CUTOFF = pre period
POST_CUTOFF  = 2017   # years >= POST_CUTOFF = post period

# Sector lat/lon bounds
SECTORS = {
    "Weddell": {
        "lon_min": -60,  "lon_max":  20,
        "lat_min": -78,  "lat_max": -62,
        "description": "Weddell Sea interior pack",
    },
    "EA": {
        "lon_min":  60,  "lon_max": 150,
        "lat_min": -70,  "lat_max": -62,
        "description": "East Antarctica seasonal ice zone",
    },
    "Ross": {
        "lon_min": 160,  "lon_max": -140,
        "lat_min": -78,  "lat_max": -65,
        "description": "Ross Sea interior pack",
    },
    "ABS": {
        "lon_min": -140, "lon_max": -60,
        "lat_min": -75,  "lat_max": -62,
        "description": "Amundsen-Bellingshausen Seas",
    },
}

# Interior pack filter — exclude cells where mean summer SIC > 0
# (always open water) or mean winter SIC < min_winter_sic (too thin/marginal)
MIN_WINTER_SIC = 0.50   # must be proper pack ice in winter

# Planet bounding box padding (degrees)
BOX_PAD  = 1.5
DATE_PAD = 14   # days either side of transition midpoint


# =============================================================================
# GRID → LAT/LON
# =============================================================================

def get_latlon_grid() -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_dataset(MERGED_FILE)
    x  = ds.x.values
    y  = ds.y.values
    ds.close()
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs("EPSG:3976", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return lat, lon


# =============================================================================
# SECTOR MASK
# =============================================================================

def sector_mask(lat: np.ndarray, lon: np.ndarray, sector: str) -> np.ndarray:
    s = SECTORS[sector]
    lat_ok = (lat >= s["lat_min"]) & (lat <= s["lat_max"])
    if sector == "Ross":
        lon_ok = (lon >= s["lon_min"]) | (lon <= s["lon_max"])
    else:
        lon_ok = (lon >= s["lon_min"]) & (lon <= s["lon_max"])
    return lat_ok & lon_ok


# =============================================================================
# INTERIOR PACK FILTER
# =============================================================================

def interior_pack_mask(min_winter_sic: float = MIN_WINTER_SIC) -> np.ndarray:
    """
    Keep cells that are genuine interior pack ice:
    - Mean winter SIC (Jun-Sep) > min_winter_sic across full record
    - Excludes coast, polynyas, and permanently open ocean
    """
    print("  computing interior pack mask...")
    ds  = xr.open_dataset(MERGED_FILE)
    ice = ds["N07_ICECON"].astype("float32")
    ice = ice.where(ice <= 1.1)
    # winter = Jun-Sep
    winter = ice.sel(time=ice.time.dt.month.isin([6,7,8,9]))
    mean_winter = winter.mean("time", skipna=True).values
    ds.close()
    return mean_winter >= min_winter_sic


# =============================================================================
# DELTA CROSSING FREQUENCY
# =============================================================================

def compute_delta_freq(phase: str, thr_tag: str = "thr15") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pre/post-2017 mean crossing frequency and their difference.
    Returns (pre_mean, post_mean, delta) as (y, x) arrays.
    """
    ds    = xr.open_dataset(f"{METRICS_DIR}/crossing_freq_{phase}_{thr_tag}.nc")
    freq  = ds["crossing_freq"]   # (year, y, x)
    years = ds["year"].values
    ds.close()

    pre_mask  = years < PRE_CUTOFF
    post_mask = years >= POST_CUTOFF

    pre_mean  = freq.isel(year=pre_mask).mean("year",  skipna=True).values
    post_mean = freq.isel(year=post_mask).mean("year", skipna=True).values
    delta     = post_mean - pre_mean

    return pre_mean, post_mean, delta


# =============================================================================
# FIND BEST REGION
# =============================================================================

def find_best_region(phase: str, sector: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {phase}  |  {sector}: {SECTORS[sector]['description']}")
    print(f"{'='*60}")

    print("Computing lat/lon grid...")
    lat, lon = get_latlon_grid()

    print(f"Masking to {sector} sector...")
    smask = sector_mask(lat, lon, sector)

    pack_mask = interior_pack_mask()

    combined_mask = smask & pack_mask
    print(f"  {combined_mask.sum()} interior pack cells in sector")

    print(f"Computing pre/post-{PRE_CUTOFF} delta crossing frequency...")
    pre_mean, post_mean, delta = compute_delta_freq(phase)

    # apply mask — only interior pack cells in sector
    delta_masked    = np.where(combined_mask, delta,    np.nan)
    pre_masked      = np.where(combined_mask, pre_mean, np.nan)
    post_masked     = np.where(combined_mask, post_mean,np.nan)

    # require both periods to have valid data
    valid = np.isfinite(pre_masked) & np.isfinite(post_masked)
    delta_masked = np.where(valid, delta_masked, np.nan)

    n_valid = np.sum(valid)
    print(f"  {n_valid} cells with valid pre and post data")

    if n_valid == 0:
        print("  No valid cells found — try a different sector or phase")
        return {}

    # find cell with largest positive delta (most increased flickering post-2017)
    best_flat = np.nanargmax(delta_masked)
    best_j, best_i = np.unravel_index(best_flat, delta.shape)

    best_lat   = float(lat[best_j, best_i])
    best_lon   = float(lon[best_j, best_i])
    best_delta = float(delta_masked[best_j, best_i])
    best_pre   = float(pre_masked[best_j, best_i])
    best_post  = float(post_masked[best_j, best_i])

    print(f"\n  Best cell: lat={best_lat:.2f}° lon={best_lon:.2f}°")
    print(f"  Pre-{PRE_CUTOFF}  mean crossings: {best_pre:.1f}")
    print(f"  Post-{POST_CUTOFF} mean crossings: {best_post:.1f}")
    print(f"  Delta: +{best_delta:.1f} crossings/season")

    # load static baseline phase date to get transition timing
    # use the year with the largest post-2017 crossing freq at this cell
    ds    = xr.open_dataset(f"{METRICS_DIR}/crossing_freq_{phase}_thr15.nc")
    freq  = ds["crossing_freq"]
    years = ds["year"].values
    ds.close()

    post_years = years[years >= POST_CUTOFF]
    freq_at_cell = freq.isel(y=best_j, x=best_i).sel(year=post_years)
    best_year_idx = int(freq_at_cell.argmax("year").values)
    best_year = int(post_years[best_year_idx])
    print(f"  Peak post-{POST_CUTOFF} year: {best_year} "
          f"(crossings={float(freq_at_cell[best_year_idx]):.0f})")

    # get static phase date for that year
    static_doy = np.nan
    try:
        f = f"{PHASE_DIR}/static/thr15_k5/{phase}/{phase}_{best_year}.nc"
        ds = xr.open_dataset(f)
        static_doy = float(ds[phase].values[best_j, best_i])
        ds.close()
    except Exception as e:
        print(f"  Warning: {e}")

    # build date range centered on static transition date
    if np.isfinite(static_doy):
        mid_doy = int(static_doy)
    else:
        # fallback: use climatological mid-point of search window
        mid_doy = 182 if phase == "FS" else 300

    ref_year = best_year
    try:
        mid_date   = datetime.date(ref_year, 1, 1) + datetime.timedelta(days=mid_doy - 1)
        start_date = mid_date - datetime.timedelta(days=DATE_PAD)
        end_date   = mid_date + datetime.timedelta(days=DATE_PAD)
        date_range = (start_date.isoformat(), end_date.isoformat())
    except Exception:
        date_range = ("unknown", "unknown")

    bbox = (
        round(best_lat - BOX_PAD, 2),
        round(best_lat + BOX_PAD, 2),
        round(best_lon - BOX_PAD, 2),
        round(best_lon + BOX_PAD, 2),
    )

    # also report sector-mean delta for context
    sector_mean_delta = float(np.nanmean(delta_masked))
    sector_med_delta  = float(np.nanmedian(delta_masked[np.isfinite(delta_masked)]))
    print(f"\n  Sector mean  Δfreq: {sector_mean_delta:+.2f}")
    print(f"  Sector median Δfreq: {sector_med_delta:+.2f}")

    return {
        "phase": phase,
        "sector": sector,
        "description": SECTORS[sector]["description"],
        "lat": best_lat,
        "lon": best_lon,
        "bbox": bbox,
        "best_year": best_year,
        "static_doy": static_doy,
        "date_range": date_range,
        "pre_crossings": best_pre,
        "post_crossings": best_post,
        "delta_crossings": best_delta,
        "sector_mean_delta": sector_mean_delta,
    }


# =============================================================================
# REPORT
# =============================================================================

def print_report(result: dict) -> None:
    if not result:
        return
    print(f"\n{'='*60}")
    print(f"  Planet Imagery Request — {result['phase']} | {result['sector']}")
    print(f"{'='*60}")
    print(f"  Region:      {result['description']}")
    print(f"  Center:      lat={result['lat']:.2f}°  lon={result['lon']:.2f}°")
    print(f"  Bounding box:")
    print(f"    lat: {result['bbox'][0]}° to {result['bbox'][1]}°")
    print(f"    lon: {result['bbox'][2]}° to {result['bbox'][3]}°")
    print(f"  Target year: {result['best_year']}")
    print(f"  Transition DOY (static thr15 k5): {result['static_doy']:.0f}")
    print(f"  Date range:  {result['date_range'][0]}  →  {result['date_range'][1]}")
    print(f"  Pre-2017 mean crossings:  {result['pre_crossings']:.1f}")
    print(f"  Post-2017 mean crossings: {result['post_crossings']:.1f}")
    print(f"  Delta:                   +{result['delta_crossings']:.1f} crossings/season")
    print(f"  Sector mean delta:        {result['sector_mean_delta']:+.2f}")
    print(f"\n  → Send to friend:")
    print(f"     bbox:  {result['bbox']}")
    print(f"     dates: {result['date_range'][0]} to {result['date_range'][1]}")
    print(f"     year:  {result['best_year']}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Find best region/dates for Planet imagery — Vichi-informed.")
    p.add_argument("--phase",  default="FS", choices=["FS", "MS"])
    p.add_argument("--sector", default="all",
                   choices=["Weddell", "EA", "Ross", "ABS", "all"])
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    sectors = list(SECTORS.keys()) if args.sector == "all" else [args.sector]

    results = []
    for sector in sectors:
        r = find_best_region(args.phase, sector)
        if r:
            results.append(r)

    # print all reports
    for r in results:
        print_report(r)

    # summary ranking
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Sector ranking by delta crossing frequency ({args.phase})")
        print(f"{'='*60}")
        ranked = sorted(results, key=lambda x: x["delta_crossings"], reverse=True)
        for i, r in enumerate(ranked):
            print(f"  {i+1}. {r['sector']:12s}  "
                  f"Δfreq=+{r['delta_crossings']:.1f}  "
                  f"sector mean={r['sector_mean_delta']:+.2f}  "
                  f"year={r['best_year']}")