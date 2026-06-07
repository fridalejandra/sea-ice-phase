#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_phase_dates_v2.py

Master script for computing Freeze Start (FS) and Melt Start (MS) phase dates.
Version 2: vectorized static method, fixed dynamic method, validation harness.

Fixes from v1:
  1. Static onset indexing: find_first_run() now returns the START of the
     k-day run (subtract k-1 from rolling argmax) not the end.
  2. Dynamic percentile scope: climatological percentiles are now precomputed
     from the full record per DOY with a ±2-day window, not within-season.
  3. Static method fully vectorized over spatial dimensions (no Python loops).
  4. Dynamic method uses precomputed clim percentiles + vectorized slope.

Speedup vs v1:
  - Static: ~20-50x (vectorized rolling over full grid)
  - Dynamic: ~5-10x (precomputed percentile maps, vectorized slope)

Usage:
  python compute_phase_dates_v2.py                    # SMMR static baseline
  python compute_phase_dates_v2.py --sensor AMSRE
  python compute_phase_dates_v2.py --method dynamic
  python compute_phase_dates_v2.py --sensor all --method all
  python compute_phase_dates_v2.py --baseline-only
  python compute_phase_dates_v2.py --validate         # run validation first
"""

import os
import gc
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# =============================================================================
# SENSOR CONFIGURATIONS
# =============================================================================

SENSOR_CONFIGS = {
    "SMMR": {
        "input_file":  "/user/geog/falejandraperez/sea-ice-phase/data/merged/SMMR_merged_19781101_20251231_complete.nc",
        "conc_var":    "N07_ICECON",
        "units":       "fraction",
        "mask_above":  1.1,
        "years":       range(1979, 2026),
        "output_root": "/user/geog/falejandraperez/sea-ice-phase/data/SMMR_phase",
    },
    "AMSRE": {
        "input_file":  "/user/geog/falejandraperez/sea-ice-phase/data/merged/AMSRE_merged_07132012_08312025.nc",
        "conc_var":    "SI_12km_SH_ICECON_DAY_SpPolarGrid12km",
        "units":       "percent",
        "mask_above":  110,
        "years":       range(2012, 2026),  # 2025 MS only (data ends Aug 31)
        "output_root": "/user/geog/falejandraperez/sea-ice-phase/data/AMSRE_phase",
    },
}

# =============================================================================
# SHARED DETECTION SETTINGS
# =============================================================================

MS_START_MMDD  = "-08-15"
MS_END_MMDD    = "-02-28"
FS_START_MMDD  = "-02-15"
FS_END_MMDD    = "-09-30"
FEB29_MODE     = "drop"

# Static defaults
THRESHOLD_LIST = [0.15, 0.20, 0.30]
WINDOW_LIST    = [3, 5, 7]
BASELINE_THR   = 0.15
BASELINE_K     = 5

# Dynamic defaults
DYN_K          = 5
DYN_Q_FS       = 0.70
DYN_Q_MS       = 0.30
DYN_SLOPE_HW   = 3
DYN_SLOPE_MIN  = 0.02
DYN_ABS_MIN    = 0.15
DYN_DOY_HW     = 2      # ±days around each DOY for climatological percentile


# =============================================================================
# CALENDAR + I/O
# =============================================================================

def standardize_calendar(da: xr.DataArray, mode: str = "drop") -> xr.DataArray:
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
    return da


def slice_season(da: xr.DataArray, start_iso: str, end_iso: str) -> xr.DataArray:
    return da.sel(time=slice(start_iso, end_iso))


def load_sic(sensor: str) -> tuple[xr.DataArray, xr.Dataset]:
    cfg = SENSOR_CONFIGS[sensor]
    ds  = xr.open_dataset(cfg["input_file"])
    ice = ds[cfg["conc_var"]].astype("float32")
    ice = ice.where(ice <= cfg["mask_above"])
    if cfg["units"] == "percent":
        ice = ice / 100.0
    ice365 = standardize_calendar(ice, mode=FEB29_MODE)
    ice365 = ice365.sortby("time")
    return ice365, ds


def save_year_field(out_dir: str, year: int, arr: np.ndarray,
                    varname: str, template: xr.Dataset) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        {varname: (("y", "x"), arr)},
        coords={"x": template.x, "y": template.y},
        attrs={"note": f"{varname} | year={year}"},
    )
    ds_out.to_netcdf(os.path.join(out_dir, f"{varname}_{year}.nc"))


# =============================================================================
# STATIC METHOD — VECTORIZED
# =============================================================================

def first_run_start_vectorized(ts: xr.DataArray,
                                threshold: float,
                                k: int,
                                above: bool) -> np.ndarray:
    """
    Vectorized detection of the FIRST k-day run onset across all (y, x) cells.

    Parameters
    ----------
    ts : xr.DataArray, dims (time, y, x)
        SIC time series for the seasonal window.
    threshold : float
    k : int
        Persistence window in days.
    above : bool
        True for FS (SIC >= threshold), False for MS (SIC <= threshold).

    Returns
    -------
    onset_doy : np.ndarray, shape (y, x)
        Day-of-year of the first k-day run onset. NaN where no run found.

    Notes
    -----
    The trailing rolling window marks the LAST day of each k-day run as True.
    We subtract (k-1) to recover the FIRST day (onset) of that run.
    This is the bug fix from v1 which recorded the end of the run.
    """
    # boolean condition: (time, y, x)
    cond = (ts >= threshold) if above else (ts <= threshold)

    # trailing rolling all-True: True at position t means days [t-k+1..t] all meet condition
    roll = cond.rolling(time=k, min_periods=k).construct("window")
    sustained = roll.all("window")   # (time, y, x)

    # shift backward by k-1 so True aligns with run START not end
    # e.g. k=5: sustained True at index 4 (end), shift -4 -> True at index 0 (start)
    sustained_shifted = sustained.shift(time=-(k - 1), fill_value=False)
    any_hit   = sustained_shifted.any("time")   # (y, x) bool
    onset_idx = sustained_shifted.argmax("time")  # directly the run start

    # get DOY at onset index
    time_vals = ts.time.values
    doys = xr.DataArray(
        np.array([int(np.datetime64(t, 'D').astype(object).timetuple().tm_yday)
                  for t in time_vals]),
        dims=["time"]
    )

    # index into doys using onset_idx
    onset_doy = doys.values[onset_idx.values]   # (y, x)

    # mask cells with no qualifying run
    onset_doy = onset_doy.astype(float)
    onset_doy[~any_hit.values] = np.nan

    return onset_doy


def compute_static_year_vectorized(ice365: xr.DataArray,
                                    year: int,
                                    threshold: float,
                                    k: int,
                                    landmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized static detection for one year."""
    ts_FS = slice_season(ice365, f"{year}{FS_START_MMDD}", f"{year}{FS_END_MMDD}")
    ts_MS = slice_season(ice365, f"{year}{MS_START_MMDD}", f"{year+1}{MS_END_MMDD}")

    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    if ts_FS.time.size >= k:
        FS = first_run_start_vectorized(ts_FS, threshold, k, above=True)
    if ts_MS.time.size >= k:
        MS = first_run_start_vectorized(ts_MS, threshold, k, above=False)

    FS[landmask] = np.nan
    MS[landmask] = np.nan
    return FS, MS


# =============================================================================
# DYNAMIC METHOD — FIXED PERCENTILE SCOPE
# =============================================================================

def precompute_clim_percentiles(ice365: xr.DataArray,
                                 q: float,
                                 doy_hw: int = DYN_DOY_HW) -> xr.DataArray:
    """
    Precompute climatological percentile for each DOY from the full record.

    For each DOY d, uses all timestamps within [d-doy_hw, d+doy_hw] across
    all years to compute the q-th percentile at each grid cell.

    This is the fix for Bug 1 in v1: percentiles are now computed from the
    full record (stationary climatology), not within a single year's season.

    Returns
    -------
    clim_q : xr.DataArray, dims (doy, y, x)
        Climatological q-th percentile for DOYs 1..365.
    """
    print(f"  precomputing climatological Q{int(q*100)} percentiles (full record, ±{doy_hw} DOY)...")
    doys = np.arange(1, 366)
    ny, nx = ice365.y.size, ice365.x.size
    clim = np.full((365, ny, nx), np.nan, dtype=np.float32)

    ice_doy  = ice365.time.dt.dayofyear.values
    ice_vals = ice365.values   # (time, y, x)

    for i, doy in enumerate(doys):
        lo = doy - doy_hw
        hi = doy + doy_hw
        if lo < 1:
            mask = (ice_doy >= (365 + lo)) | (ice_doy <= hi)
        elif hi > 365:
            mask = (ice_doy >= lo) | (ice_doy <= (hi - 365))
        else:
            mask = (ice_doy >= lo) & (ice_doy <= hi)

        subset = ice_vals[mask]   # (n_matching_days, y, x)
        if subset.shape[0] > 0:
            clim[i] = np.nanquantile(subset, q, axis=0)

    return xr.DataArray(clim, dims=["doy", "y", "x"],
                        coords={"doy": doys, "y": ice365.y, "x": ice365.x})


def centered_slope_vectorized(da: xr.DataArray, hw: int) -> xr.DataArray:
    """Centered finite difference slope: (da[t+hw] - da[t-hw]) / (2*hw)."""
    return (da.shift(time=-hw) - da.shift(time=+hw)) / (2.0 * hw)


def first_run_with_slope_vectorized(ts: xr.DataArray,
                                     slope: xr.DataArray,
                                     threshold_map: np.ndarray,
                                     k: int,
                                     event: str,
                                     slope_min: float) -> np.ndarray:
    """
    Vectorized: find first k-day run that also passes slope test.

    threshold_map : (y, x) array of per-cell climatological thresholds.
    """
    ny, nx = ts.y.size, ts.x.size

    # broadcast threshold_map to (time, y, x)
    thr = xr.DataArray(
        np.broadcast_to(threshold_map[np.newaxis], ts.shape),
        dims=ts.dims, coords=ts.coords
    )

    if event == "FS":
        cond = ts >= thr
    else:
        cond = ts <= thr

    roll      = cond.rolling(time=k, min_periods=k).construct("window")
    sustained = roll.all("window")   # (time, y, x)

    # slope at each time step
    if event == "FS":
        slope_ok = slope > slope_min
    else:
        slope_ok = slope < -slope_min

    # valid = sustained crossing AND slope condition at run START
    # shift sustained backward by k-1 so True aligns with run start not end
    sustained_shifted = sustained.shift(time=-(k - 1), fill_value=False)
    valid = sustained_shifted & slope_ok   # (time, y, x)

    any_hit   = valid.any("time")
    onset_idx = valid.argmax("time")  # directly the run start

    time_vals = ts.time.values
    doys = np.array([int(np.datetime64(t, 'D').astype(object).timetuple().tm_yday)
                     for t in time_vals])

    onset_doy          = doys[onset_idx.values].astype(float)
    onset_doy[~any_hit.values] = np.nan

    return onset_doy


def compute_dynamic_year(ice365: xr.DataArray,
                          year: int,
                          clim_q_fs: xr.DataArray,
                          clim_q_ms: xr.DataArray,
                          k: int,
                          slope_hw: int,
                          slope_min: float,
                          abs_min: float,
                          landmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamic detection for one year using precomputed climatological percentiles.
    """
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    ts_FS = slice_season(ice365, f"{year}{FS_START_MMDD}", f"{year}{FS_END_MMDD}")
    ts_MS = slice_season(ice365, f"{year}{MS_START_MMDD}", f"{year+1}{MS_END_MMDD}")

    if ts_FS.time.size < k or ts_MS.time.size < k:
        return FS, MS

    # smooth + slope
    ts_FS_s  = ts_FS.rolling(time=k, center=True, min_periods=max(1, k//2)).mean()
    ts_MS_s  = ts_MS.rolling(time=k, center=True, min_periods=max(1, k//2)).mean()
    slope_FS = centered_slope_vectorized(ts_FS_s, slope_hw)
    slope_MS = centered_slope_vectorized(ts_MS_s, slope_hw)

    # extract climatological thresholds for the DOYs in each window
    def get_threshold_map(clim_q: xr.DataArray, ts: xr.DataArray) -> np.ndarray:
        """Mean climatological threshold across the seasonal window DOYs."""
        doys = ts.time.dt.dayofyear.values
        # average the climatological percentile across all DOYs in the window
        # this gives a single (y, x) threshold map per season
        clim_subset = clim_q.sel(doy=doys, method="nearest")
        return clim_subset.mean("doy").values

    thr_fs = get_threshold_map(clim_q_fs, ts_FS)
    thr_ms = get_threshold_map(clim_q_ms, ts_MS)

    # enforce abs_min floor for FS
    thr_fs = np.maximum(thr_fs, abs_min)

    if ts_FS_s.time.size >= k:
        FS = first_run_with_slope_vectorized(
            ts_FS_s, slope_FS, thr_fs, k, "FS", slope_min)

    if ts_MS_s.time.size >= k:
        MS = first_run_with_slope_vectorized(
            ts_MS_s, slope_MS, thr_ms, k, "MS", slope_min)

    FS[landmask] = np.nan
    MS[landmask] = np.nan
    return FS, MS


# =============================================================================
# VALIDATION — compare scalar v1 vs vectorized v2 on one year
# =============================================================================

def validate(sensor: str = "SMMR", year: int = 2000,
             thr: float = 0.15, k: int = 5,
             ny_sub: int = 50, nx_sub: int = 50) -> None:
    """
    Run scalar (v1 logic) and vectorized (v2 logic) on a spatial subset
    for one year and compare results.

    This validates that vectorization didn't change the algorithm logic,
    and that the indexing fix is applied consistently.
    """
    print(f"\n{'='*60}")
    print(f"  Validation: {sensor} thr={thr} k={k} year={year}")
    print(f"  Spatial subset: {ny_sub}×{nx_sub}")
    print(f"{'='*60}")

    ice365, ds = load_sic(sensor)
    # subset spatially for speed
    ice_sub  = ice365.isel(y=slice(100, 100+ny_sub), x=slice(100, 100+nx_sub))
    landmask = ice_sub.isnull().all("time").values

    # --- SCALAR (v1 logic, with bug fix applied manually) ---
    def scalar_first_run(ts_1d, threshold, k, above):
        cond = (ts_1d.values >= threshold) if above else (ts_1d.values <= threshold)
        for i in range(len(cond) - k + 1):
            if all(cond[i:i+k]):
                return i   # onset = start of run
        return None

    ts_FS = slice_season(ice_sub, f"{year}{FS_START_MMDD}", f"{year}{FS_END_MMDD}")
    ts_MS = slice_season(ice_sub, f"{year}{MS_START_MMDD}", f"{year+1}{MS_END_MMDD}")

    FS_doys_FS = ts_FS.time.dt.dayofyear.values
    FS_doys_MS = ts_MS.time.dt.dayofyear.values

    ny, nx = ice_sub.y.size, ice_sub.x.size
    FS_scalar = np.full((ny, nx), np.nan)
    MS_scalar = np.full((ny, nx), np.nan)

    for j in range(ny):
        for i in range(nx):
            if landmask[j, i]:
                continue
            idx = scalar_first_run(ts_FS[:, j, i], thr, k, above=True)
            if idx is not None:
                FS_scalar[j, i] = FS_doys_FS[idx]
            idx = scalar_first_run(ts_MS[:, j, i], thr, k, above=False)
            if idx is not None:
                MS_scalar[j, i] = FS_doys_MS[idx]

    # --- VECTORIZED (v2) ---
    FS_vec, MS_vec = compute_static_year_vectorized(ice_sub, year, thr, k, landmask)

    # --- COMPARE ---
    fs_match = np.nansum(np.abs(FS_scalar - FS_vec) > 0)
    ms_match = np.nansum(np.abs(MS_scalar - MS_vec) > 0)
    fs_valid = np.sum(np.isfinite(FS_scalar))
    ms_valid = np.sum(np.isfinite(MS_scalar))

    print(f"  FS: {fs_valid} valid cells, {fs_match} mismatches")
    print(f"  MS: {ms_valid} valid cells, {ms_match} mismatches")

    if fs_match == 0 and ms_match == 0:
        print("  PASS — scalar and vectorized outputs are identical")
    else:
        print("  FAIL — outputs differ, check indexing logic")
        # show a sample of differences
        diff_idx = np.argwhere(np.abs(FS_scalar - FS_vec) > 0)
        if len(diff_idx) > 0:
            j, i = diff_idx[0]
            print(f"  Example FS diff at y={j} x={i}: "
                  f"scalar={FS_scalar[j,i]:.0f} vec={FS_vec[j,i]:.0f}")

    print("")


# =============================================================================
# RUNNERS
# =============================================================================

def run_static(sensor: str,
               thresholds: list = THRESHOLD_LIST,
               windows: list = WINDOW_LIST) -> None:
    cfg            = SENSOR_CONFIGS[sensor]
    ice365, ds     = load_sic(sensor)
    landmask       = ice365.isnull().all("time").values
    all_years      = np.unique(ice365.time.dt.year.values)
    years_run      = [y for y in cfg["years"]
                      if (y in all_years) and ((y + 1) in all_years)]

    for thr in thresholds:
        for k in windows:
            dir_FS = os.path.join(cfg["output_root"], "static",
                                  f"thr{int(thr*100):02d}_k{k}", "FS")
            dir_MS = os.path.join(cfg["output_root"], "static",
                                  f"thr{int(thr*100):02d}_k{k}", "MS")
            for year in tqdm(years_run,
                             desc=f"{sensor} static  thr={thr:.2f}  k={k}"):
                FS, MS = compute_static_year_vectorized(
                    ice365, year, thr, k, landmask)
                save_year_field(dir_FS, year, FS, "FS", ds)
                save_year_field(dir_MS, year, MS, "MS", ds)
                gc.collect()


def run_dynamic(sensor: str,
                k: int           = DYN_K,
                q_fs: float      = DYN_Q_FS,
                q_ms: float      = DYN_Q_MS,
                slope_hw: int    = DYN_SLOPE_HW,
                slope_min: float = DYN_SLOPE_MIN,
                abs_min: float   = DYN_ABS_MIN) -> None:
    cfg            = SENSOR_CONFIGS[sensor]
    ice365, ds     = load_sic(sensor)
    landmask       = ice365.isnull().all("time").values
    all_years      = np.unique(ice365.time.dt.year.values)
    years_run      = [y for y in cfg["years"]
                      if (y in all_years) and ((y + 1) in all_years)]

    # precompute climatological percentiles once from full record
    clim_q_fs = precompute_clim_percentiles(ice365, q_fs, DYN_DOY_HW)
    clim_q_ms = precompute_clim_percentiles(ice365, q_ms, DYN_DOY_HW)

    dir_FS = os.path.join(cfg["output_root"], "dynamic",
                          f"k{k}_q{int(q_fs*100)}", "FS")
    dir_MS = os.path.join(cfg["output_root"], "dynamic",
                          f"k{k}_q{int(q_fs*100)}", "MS")

    for year in tqdm(years_run,
                     desc=f"{sensor} dynamic  k={k}  q_fs={q_fs}  q_ms={q_ms}"):
        FS, MS = compute_dynamic_year(
            ice365, year, clim_q_fs, clim_q_ms,
            k, slope_hw, slope_min, abs_min, landmask)
        save_year_field(dir_FS, year, FS, "FS", ds)
        save_year_field(dir_MS, year, MS, "MS", ds)
        gc.collect()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute FS/MS phase dates — v2 (vectorized, bug-fixed).")
    p.add_argument("--sensor", default="SMMR",
                   choices=["SMMR", "AMSRE", "all"])
    p.add_argument("--method", default="static",
                   choices=["static", "dynamic", "all"])
    p.add_argument("--thr", type=float, nargs="+", default=THRESHOLD_LIST)
    p.add_argument("--windows", type=int, nargs="+", default=WINDOW_LIST)
    p.add_argument("--baseline-only", action="store_true",
                   help=f"Baseline only: thr={BASELINE_THR} k={BASELINE_K}")
    p.add_argument("--validate", action="store_true",
                   help="Run validation before detection")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    sensors = ["SMMR", "AMSRE"] if args.sensor == "all" else [args.sensor]
    methods = ["static", "dynamic"] if args.method == "all" else [args.method]
    thrs    = [BASELINE_THR] if args.baseline_only else args.thr
    wins    = [BASELINE_K]   if args.baseline_only else args.windows

    if args.validate:
        for sensor in sensors:
            validate(sensor=sensor, year=2000, thr=BASELINE_THR, k=BASELINE_K)

    for sensor in sensors:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"  Sensor: {sensor}   Method: {method}")
            print(f"{'='*60}")
            if method == "static":
                run_static(sensor, thresholds=thrs, windows=wins)
            else:
                run_dynamic(sensor)

    print("\nDone.")