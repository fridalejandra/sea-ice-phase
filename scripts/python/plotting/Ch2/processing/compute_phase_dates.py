#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_phase_dates.py

Master script for computing Freeze Start (FS) and Melt Start (MS) phase dates.
Covers all four combinations:

  Sensor  × Method
  ──────────────────────────────────────────────────────────────────
  SMMR    × static   — fixed threshold + k-day persistence
  SMMR    × dynamic  — percentile threshold + slope filter
  AMSR-E  × static   — fixed threshold + k-day persistence
  AMSR-E  × dynamic  — percentile threshold + slope filter

Outputs (NetCDF, one file per year):
  results/{SENSOR}_phase/static/FS_thrXX_kY/FS_YYYY.nc
  results/{SENSOR}_phase/static/MS_thrXX_kY/MS_YYYY.nc
  results/{SENSOR}_phase/dynamic/quantile_k{K}/FS/p{Q}/FS_YYYY.nc
  results/{SENSOR}_phase/dynamic/quantile_k{K}/MS/p{Q}/MS_YYYY.nc

Usage:
  python compute_phase_dates.py                          # SMMR static baseline
  python compute_phase_dates.py --sensor AMSRE
  python compute_phase_dates.py --method dynamic
  python compute_phase_dates.py --sensor AMSRE --method dynamic
  python compute_phase_dates.py --sensor all --method all   # all four
  python compute_phase_dates.py --baseline-only             # thr=0.15 k=5 only
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
        "input_file":  "/user/geog/falejandraperez/sea-ice-phase/data/merged/SMMR_merged_19781101_20251231.nc",
        "conc_var":    "N07_ICECON",
        "units":       "fraction",       # 0-1
        "mask_above":  1.1,
        "years":       range(1979, 2026),  # full years 1979-2025
        "output_root": "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase",
    },
    "AMSRE": {
        "input_file":  "/user/geog/falejandraperez/sea-ice-phase/data/merged/AMSRE_merged_07132012_08312025.nc",
        "conc_var":    "SI_12km_SH_ICECON_DAY_SpPolarGrid12km",
        "units":       "percent",        # 0-100, auto-converted to fraction
        "mask_above":  110,
        "years":       range(2012, 2026),  # full years 2012-2024; 2025 MS only (data ends Aug 31)
        "output_root": "/user/geog/falejandraperez/sea-ice-phase/results/AMSRE_phase",
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
THRESHOLD_LIST = [0.15, 0.20, 0.30]   # 15% is the baseline minimum
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


# =============================================================================
# CALENDAR + I/O HELPERS
# =============================================================================

def standardize_calendar(da: xr.DataArray, mode: str = "drop") -> xr.DataArray:
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
    return da


def slice_season(da: xr.DataArray, start_iso: str, end_iso: str) -> xr.DataArray:
    return da.sel(time=slice(start_iso, end_iso))


def load_sic(sensor: str) -> tuple[xr.DataArray, xr.Dataset]:
    """Load, scale, mask, and calendar-standardize SIC for a given sensor."""
    cfg = SENSOR_CONFIGS[sensor]
    ds  = xr.open_dataset(cfg["input_file"])
    ice = ds[cfg["conc_var"]].astype("float32")
    ice = ice.where(ice <= cfg["mask_above"])
    if cfg["units"] == "percent":
        ice = ice / 100.0
    ice365 = standardize_calendar(ice, mode=FEB29_MODE)
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
# DETECTION PRIMITIVES
# =============================================================================

def find_first_run(ts: xr.DataArray, threshold: float,
                   k: int, above: bool) -> int | float:
    cond = (ts >= threshold) if above else (ts <= threshold)
    roll = cond.rolling(time=k).construct("window")
    hits = roll.all("window")
    if not bool(hits.any()):
        return np.nan
    return int(hits.argmax("time").item())


def rolling_mean_centered(da: xr.DataArray, k: int) -> xr.DataArray:
    return da.rolling(time=k, center=True, min_periods=max(1, k // 2)).mean()


def centered_slope(da: xr.DataArray, hw: int) -> xr.DataArray:
    return (da.shift(time=-hw) - da.shift(time=+hw)) / (2.0 * hw)


def first_run_with_slope(ts: xr.DataArray, slope: xr.DataArray,
                         threshold: float, k: int, event: str,
                         slope_min: float) -> int | float:
    """First k-day sustained threshold crossing that passes the slope test."""
    if event == "FS":
        side     = ts >= threshold
        slope_ok = lambda s: s > +slope_min
    else:
        side     = ts <= threshold
        slope_ok = lambda s: s < -slope_min

    if not bool(side.any()):
        return np.nan

    roll      = side.rolling(time=k).construct("window")
    sustained = roll.all("window")
    t_idxs    = np.where(sustained.values)[0]
    if t_idxs.size == 0:
        return np.nan

    start_idxs = t_idxs - (k - 1)
    start_idxs = start_idxs[start_idxs >= 0]
    for sidx in start_idxs:
        s_val = float(slope.isel(time=int(sidx)).values)
        if np.isfinite(s_val) and slope_ok(s_val):
            return int(sidx)
    return np.nan


# =============================================================================
# PER-YEAR COMPUTATION
# =============================================================================

def compute_static_year(ice365: xr.DataArray, year: int,
                         threshold: float, k: int,
                         landmask: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    ts_MS = slice_season(ice365, f"{year}{MS_START_MMDD}", f"{year+1}{MS_END_MMDD}")
    ts_FS = slice_season(ice365, f"{year}{FS_START_MMDD}", f"{year}{FS_END_MMDD}")

    if ts_MS.time.size < max(60, k) and ts_FS.time.size < max(60, k):
        return FS, MS

    for j in range(ny):
        col_MS = ts_MS.isel(y=j).transpose("time", "x")
        col_FS = ts_FS.isel(y=j).transpose("time", "x")
        for i in range(nx):
            if bool(landmask.isel(y=j, x=i)):
                continue
            ts_r = col_MS[:, i]
            ts_a = col_FS[:, i]
            if bool((ts_r <= threshold).any()):
                idx = find_first_run(ts_r, threshold, k, above=False)
                if not np.isnan(idx):
                    MS[j, i] = ts_r.time[int(idx)].dt.dayofyear.item()
            if bool((ts_a >= threshold).any()):
                idx = find_first_run(ts_a, threshold, k, above=True)
                if not np.isnan(idx):
                    FS[j, i] = ts_a.time[int(idx)].dt.dayofyear.item()

    FS[landmask.values] = np.nan
    MS[landmask.values] = np.nan
    return FS, MS


def compute_dynamic_year(ice365: xr.DataArray, year: int,
                          k: int, q_fs: float, q_ms: float,
                          slope_hw: int, slope_min: float, abs_min: float,
                          landmask: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    ts_MS = slice_season(ice365, f"{year}{MS_START_MMDD}", f"{year+1}{MS_END_MMDD}")
    ts_FS = slice_season(ice365, f"{year}{FS_START_MMDD}", f"{year}{FS_END_MMDD}")

    if ts_MS.time.size < max(60, k) and ts_FS.time.size < max(60, k):
        return FS, MS

    ts_MS_s  = rolling_mean_centered(ts_MS, k)
    ts_FS_s  = rolling_mean_centered(ts_FS, k)
    slope_MS = centered_slope(ts_MS_s, slope_hw)
    slope_FS = centered_slope(ts_FS_s, slope_hw)

    for j in range(ny):
        col_MS_s  = ts_MS_s.isel(y=j).transpose("time", "x")
        col_FS_s  = ts_FS_s.isel(y=j).transpose("time", "x")
        col_sl_MS = slope_MS.isel(y=j).transpose("time", "x")
        col_sl_FS = slope_FS.isel(y=j).transpose("time", "x")

        for i in range(nx):
            if bool(landmask.isel(y=j, x=i)):
                continue

            ts_r = col_MS_s[:, i]
            sl_r = col_sl_MS[:, i]
            ts_a = col_FS_s[:, i]
            sl_a = col_sl_FS[:, i]

            # local climatological percentile thresholds
            thr_ms     = float(ts_r.quantile(q_ms).values)
            thr_fs_eff = max(float(ts_a.quantile(q_fs).values), abs_min)

            if bool(np.isfinite(ts_r).any()):
                idx = first_run_with_slope(ts_r, sl_r, thr_ms, k,
                                           event="MS", slope_min=slope_min)
                if not np.isnan(idx):
                    MS[j, i] = ts_r.time[int(idx)].dt.dayofyear.item()

            if bool(np.isfinite(ts_a).any()):
                idx = first_run_with_slope(ts_a, sl_a, thr_fs_eff, k,
                                           event="FS", slope_min=slope_min)
                if not np.isnan(idx):
                    FS[j, i] = ts_a.time[int(idx)].dt.dayofyear.item()

    FS[landmask.values] = np.nan
    MS[landmask.values] = np.nan
    return FS, MS


# =============================================================================
# RUNNERS
# =============================================================================

def run_static(sensor: str,
               thresholds: list = THRESHOLD_LIST,
               windows: list = WINDOW_LIST) -> None:
    cfg            = SENSOR_CONFIGS[sensor]
    ice365, ds     = load_sic(sensor)
    landmask       = ice365.isnull().all("time")
    all_years      = np.unique(ice365.time.dt.year.values)
    years_run      = [y for y in cfg["years"]
                      if (y in all_years) and ((y + 1) in all_years)]

    for thr in thresholds:
        for k in windows:
            dir_FS = os.path.join(cfg["output_root"], "static",
                                  f"FS_thr{int(thr*100):02d}_k{k}")
            dir_MS = os.path.join(cfg["output_root"], "static",
                                  f"MS_thr{int(thr*100):02d}_k{k}")
            for year in tqdm(years_run,
                             desc=f"{sensor} static  thr={thr:.2f}  k={k}"):
                FS, MS = compute_static_year(ice365, year, thr, k, landmask)
                save_year_field(dir_FS, year, FS, "FS", ds)
                save_year_field(dir_MS, year, MS, "MS", ds)
                gc.collect()


def run_dynamic(sensor: str,
                k: int        = DYN_K,
                q_fs: float   = DYN_Q_FS,
                q_ms: float   = DYN_Q_MS,
                slope_hw: int = DYN_SLOPE_HW,
                slope_min: float = DYN_SLOPE_MIN,
                abs_min: float   = DYN_ABS_MIN) -> None:
    cfg            = SENSOR_CONFIGS[sensor]
    ice365, ds     = load_sic(sensor)
    landmask       = ice365.isnull().all("time")
    all_years      = np.unique(ice365.time.dt.year.values)
    years_run      = [y for y in cfg["years"]
                      if (y in all_years) and ((y + 1) in all_years)]

    dir_FS = os.path.join(cfg["output_root"], "dynamic",
                          f"quantile_k{k}", "FS", f"p{int(q_fs*10)}")
    dir_MS = os.path.join(cfg["output_root"], "dynamic",
                          f"quantile_k{k}", "MS", f"p{int(q_ms*10)}")

    for year in tqdm(years_run,
                     desc=f"{sensor} dynamic  k={k}  q_fs={q_fs}  q_ms={q_ms}"):
        FS, MS = compute_dynamic_year(ice365, year, k, q_fs, q_ms,
                                      slope_hw, slope_min, abs_min, landmask)
        save_year_field(dir_FS, year, FS, "FS", ds)
        save_year_field(dir_MS, year, MS, "MS", ds)
        gc.collect()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute FS/MS phase dates for SMMR and/or AMSR-E.")
    p.add_argument("--sensor", default="SMMR",
                   choices=["SMMR", "AMSRE", "all"],
                   help="Sensor to process  (default: SMMR)")
    p.add_argument("--method", default="static",
                   choices=["static", "dynamic", "all"],
                   help="Detection method   (default: static)")
    p.add_argument("--thr", type=float, nargs="+", default=THRESHOLD_LIST,
                   help="SIC threshold(s) for static, as fractions e.g. 0.15")
    p.add_argument("--windows", type=int, nargs="+", default=WINDOW_LIST,
                   help="Persistence window(s) in days for static method")
    p.add_argument("--baseline-only", action="store_true",
                   help=f"Static baseline only: thr={BASELINE_THR} k={BASELINE_K}")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    sensors = ["SMMR", "AMSRE"] if args.sensor == "all" else [args.sensor]
    methods = ["static", "dynamic"] if args.method == "all" else [args.method]
    thrs    = [BASELINE_THR] if args.baseline_only else args.thr
    wins    = [BASELINE_K]   if args.baseline_only else args.windows

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