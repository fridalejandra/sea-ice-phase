#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMMR Phase Detection (Static Method) with Smoothing + Persistence + Slope
- FS (advance): first H-day sustained run of (k-day SMOOTHED SIC) >= T inside [Feb 15, Sep 30]
- MS (retreat): first H-day sustained run of (k-day SMOOTHED SIC) <= T inside [Aug 15, Feb 28(next)]
- Persistence: H consecutive days on the new side of the threshold (H may differ from k)
- Slope check: FS requires dSIC/dt > +slope_min, MS requires dSIC/dt < -slope_min (centered ±slope_hw days)
- Calendar: standardized to 365 DOY by default (drops Feb 29)
- Land/missing masking applied once, reused every year
- Wrapper sweeps thresholds and windows (where 'k' is the smoothing window)
"""

import os, gc, calendar
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

# === CONFIGURATION === #
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"
CONC_VAR = "N07_ICECON"
YEARS = range(1979, 2025)
THRESHOLD = [0.10, 0.15, 0.30]   # SMMR is 0..1
WINDOWS = [3, 5, 7]              # used here as SMOOTHING window (k)
SENSOR_TAG = "SMMR"

# Search windows (indices are built per-year on a 365-day timeline)
MS_START_MMDD = "-08-15"  # retreat search begins Aug 15 (year y)
MS_END_MMDD   = "-02-28"  # ends Feb 28 (year y+1)
FS_START_MMDD = "-02-15"  # advance search begins Feb 15 (year y)
FS_END_MMDD   = "-09-30"  # ends Sep 30 (year y)

# Feb 29 policy
FEB29_MODE = "drop"

# >>> NEW: Detection parameters (separate from smoothing)
PERSIST_DAYS = 3          # H: required consecutive days on the new side
SLOPE_HW     = 3          # half-width (days) for centered slope derivative
SLOPE_MIN    = 0.03       # minimum slope magnitude (fraction/day) at event

# === HELPERS === #
def standardize_calendar(da: xr.DataArray, mode: str = "drop") -> xr.DataArray:
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
    elif mode == "keep":
        return da
    else:
        raise ValueError("FEB29_MODE must be 'drop' or 'keep'.")

def slice_season(da: xr.DataArray, start_iso: str, end_iso: str) -> xr.DataArray:
    return da.sel(time=slice(start_iso, end_iso))

# >>> NEW: smoothing and slope
def rolling_mean_centered(da: xr.DataArray, k: int) -> xr.DataArray:
    # centered moving mean; keep edges with minimal periods
    return da.rolling(time=k, center=True, min_periods=max(1, k//2)).mean()

def centered_slope(da_smooth: xr.DataArray, hw: int) -> xr.DataArray:
    # (SIC[t+hw] - SIC[t-hw]) / (2*hw)
    return (da_smooth.shift(time=-hw) - da_smooth.shift(time=+hw)) / (2.0 * hw)

# >>> NEW: first sustained crossing with slope test
def first_sustained_with_slope(ts_smooth: xr.DataArray,
                               slope: xr.DataArray,
                               threshold: float,
                               H: int,
                               event: str,
                               slope_min: float) -> int | float:
    """
    Return integer time-index of the first H-day sustained crossing that also
    satisfies the slope sign + magnitude at the first day of the run. Else np.nan.

    ts_smooth: (time,) SIC (smoothed) at a single pixel
    slope:     (time,) centered slope at the same pixel
    event: 'FS' (>=T) or 'MS' (<=T)
    """
    if event == "FS":
        side = ts_smooth >= threshold
        slope_ok = lambda s: (s > +slope_min)
    else:
        side = ts_smooth <= threshold
        slope_ok = lambda s: (s < -slope_min)

    if not bool(side.any()):
        return np.nan

    # rolling window of H with "all True" test
    roll = side.rolling(time=H).construct("window")
    sustained = roll.all("window")  # True starting at each window's end index

    if not bool(sustained.any()):
        return np.nan

    # We need the FIRST time index that starts an H-run AND passes slope check at its start
    # Build a mask for the start index of each qualifying run
    # 'sustained' True at t means the window [t-H+1 .. t] is all True
    # So the start index is (t-H+1)
    t_idxs = np.where(sustained.values)[0]
    if t_idxs.size == 0:
        return np.nan

    start_idxs = t_idxs - (H - 1)
    start_idxs = start_idxs[start_idxs >= 0]

    # iterate over candidates in time order; pick first with slope criterion
    for sidx in start_idxs:
        s_val = float(slope.isel(time=int(sidx)).values)
        if np.isfinite(s_val) and slope_ok(s_val):
            return int(sidx)  # index into ts_smooth.time
    return np.nan

def save_year_field(out_dir: str, year: int, arr: np.ndarray, varname: str, template: xr.Dataset):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        {varname: (("y","x"), arr)},
        coords={"x": template.x, "y": template.y},
        attrs={"note": f"{varname} | year={year}"}
    )
    ds_out.to_netcdf(os.path.join(out_dir, f"{varname}_{year}.nc"))

# =======================
# CORE PER-YEAR COMPUTATION
# =======================
def compute_FS_MS_for_year(ice365: xr.DataArray,
                           year: int,
                           threshold: float,
                           k: int,
                           landmask: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FS (advance) and MS (retreat) for one calendar year (y)
    using SMOOTHED SIC (k-day mean), H-day persistence, and slope test.

    Returns two (ny, nx) arrays of DOY (1..365) with NaN where undefined.
    """
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    # Build seasonal windows
    y, y1 = year, year + 1
    ts_MS = slice_season(ice365, f"{y}{MS_START_MMDD}", f"{y1}{MS_END_MMDD}")
    ts_FS = slice_season(ice365, f"{y}{FS_START_MMDD}", f"{y}{FS_END_MMDD}")

    # Safety
    if ts_MS.time.size < max(60, k) and ts_FS.time.size < max(60, k):
        return FS, MS

    # >>> NEW: precompute smoothed series and slope for both windows
    ts_MS_s = rolling_mean_centered(ts_MS, k)
    ts_FS_s = rolling_mean_centered(ts_FS, k)
    slope_MS = centered_slope(ts_MS_s, SLOPE_HW)
    slope_FS = centered_slope(ts_FS_s, SLOPE_HW)

    # iterate grid (explicit loops; can vectorize later)
    for j in range(ny):
        col_MS_s = ts_MS_s.isel(y=j).transpose("time", "x")  # (t,x)
        col_FS_s = ts_FS_s.isel(y=j).transpose("time", "x")
        col_sl_MS = slope_MS.isel(y=j).transpose("time", "x")
        col_sl_FS = slope_FS.isel(y=j).transpose("time", "x")
        for i in range(nx):
            if landmask[j, i]:
                continue

            ts_r = col_MS_s[:, i]    # retreat (Aug→Feb), smoothed
            sl_r = col_sl_MS[:, i]
            ts_a = col_FS_s[:, i]    # advance (Feb→Sep), smoothed
            sl_a = col_sl_FS[:, i]

            # RETREAT (MS): first H-day run <= T with negative slope at run start
            if bool(np.isfinite(ts_r).any()):
                idx_rt = first_sustained_with_slope(
                    ts_r, sl_r, threshold, H=PERSIST_DAYS, event="MS", slope_min=SLOPE_MIN
                )
                if not np.isnan(idx_rt):
                    MS[j, i] = ts_r.time[int(idx_rt)].dt.dayofyear.item()

            # ADVANCE (FS): first H-day run >= T with positive slope at run start
            if bool(np.isfinite(ts_a).any()):
                idx_ad = first_sustained_with_slope(
                    ts_a, sl_a, threshold, H=PERSIST_DAYS, event="FS", slope_min=SLOPE_MIN
                )
                if not np.isnan(idx_ad):
                    FS[j, i] = ts_a.time[int(idx_ad)].dt.dayofyear.item()

    # mask land
    FS[landmask.values] = np.nan
    MS[landmask.values] = np.nan
    return FS, MS

# =======================
# WRAPPER: RUN A PARAMETER SET
# =======================
def run_phase_detection(thr: float,
                        k: int,
                        years: list[int] | range,
                        input_file: str = INPUT_FILE,
                        conc_var: str = CONC_VAR,
                        output_root: str = OUTPUT_DIR,
                        sensor_tag: str = SENSOR_TAG,
                        feb29_mode: str = FEB29_MODE):
    """
    Wrapper: runs FS and MS for a given (threshold, k) over a list/range of years.
    Writes two folders:
      - {output_root}/FS_thr{XX}_k{K}/FS_YYYY.nc
      - {output_root}/MS_thr{XX}_k{K}/MS_YYYY.nc
    Here k = smoothing window; H, slope_min, slope_hw are global config.
    """
    ds = xr.open_dataset(input_file)[[conc_var, "x", "y", "time"]]
    ice = ds[conc_var].astype("float32")

    # Scale if needed (if native units are 0..100)
    if float(ice.max()) > 1.5:
        ice = ice / 100.0

    # mask land/missing; assume >1 is invalid if raw 0..1
    ice = ice.where((ice >= 0.0) & (ice <= 1.1))

    # standardize calendar
    ice365 = standardize_calendar(ice, mode=feb29_mode)

    # land mask: cells NA for entire year-time axis (for this file)
    landmask = ice365.isnull().all("time")

    # output dirs
    dir_FS = os.path.join(output_root, f"FS_thr{int(thr*100):02d}_k{k}")
    dir_MS = os.path.join(output_root, f"MS_thr{int(thr*100):02d}_k{k}")
    Path(dir_FS).mkdir(parents=True, exist_ok=True)
    Path(dir_MS).mkdir(parents=True, exist_ok=True)

    # ensure years exist (need y and y+1 for MS)
    all_years = np.unique(ice365.time.dt.year.values)
    years_run = [y for y in years if y in all_years and (y+1) in all_years]

    for year in tqdm(years_run, desc=f"Running {sensor_tag} thr={thr:.2f}, k={k}, H={PERSIST_DAYS}, slope≥{SLOPE_MIN}"):
        FS, MS = compute_FS_MS_for_year(ice365, year, thr, k, landmask)
        save_year_field(dir_FS, year, FS, "FS", ds)
        save_year_field(dir_MS, year, MS, "MS", ds)
        gc.collect()

# =======================
# CONVENIENCE DRIVERS
# =======================
def run_threshold_sweep(years, k=5, thresholds=(0.10, 0.15, 0.30)):
    for thr in thresholds:
        run_phase_detection(thr=thr, k=k, years=years)

def run_window_sweep(years, thr=0.15, windows=(3, 5, 7)):
    for k in windows:
        run_phase_detection(thr=thr, k=k, years=years)

# =======================
# EXAMPLE MAIN
# =======================
if __name__ == "__main__":
    YEARS = range(1979, 2025)

    # A) Threshold sensitivity at fixed k=5
    run_threshold_sweep(years=YEARS, k=5, thresholds=(0.10, 0.15, 0.30))

    # B) Window sensitivity at fixed thr=0.15
    run_window_sweep(years=YEARS, thr=0.15, windows=(3, 5, 7))

    print("Done.")
