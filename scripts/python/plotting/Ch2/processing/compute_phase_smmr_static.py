#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Static FS/MS Phase Detection (Bootstrap/SMMR-style)

- FS (advance): first k-day run with SIC >= threshold
  Search window: Feb 15 (y) → Sep 30 (y)

- MS (retreat): first k-day run with SIC <= threshold
  Search window: Aug 15 (y) → Feb 28 (y+1)

- Persistence: rolling "all True" over k days
- Calendar: standardized to 365 DOY by default (Feb 29 dropped)
- Land/missing masking applied once, reused every year
- Simple wrappers to sweep thresholds and persistence windows
"""

import os
import gc
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# =======================
# CONFIGURATION
# =======================

# Input SIC file (daily, SH, with x/y/time)
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"

# Output root directory
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/SMMR_phase"

# Name of SIC variable in INPUT_FILE
CONC_VAR = "N07_ICECON"

# Default years and parameter sets (you can override in __main__)
YEARS = range(1979, 2025)
THRESHOLD_LIST = [0.10, 0.15, 0.30]   # SIC thresholds (0–1)
WINDOW_LIST = [3, 5, 7]               # k-day persistence windows

# Simple tag for filenames
SENSOR_TAG = "SMMR"

# Search windows (Southern Hemisphere)
# MS (retreat): y-08-15 → (y+1)-02-28
# FS (advance): y-02-15 → y-09-30
MS_START_MMDD = "-08-15"
MS_END_MMDD   = "-02-28"
FS_START_MMDD = "-02-15"
FS_END_MMDD   = "-09-30"

# Feb 29 policy
#  "drop": remove Feb 29 so every year has 365 days (recommended)
#  "keep": keep Feb 29 (then you must be careful with cross-year windows)
FEB29_MODE = "drop"


# =======================
# HELPER FUNCTIONS
# =======================

def standardize_calendar(da: xr.DataArray, mode: str = "drop") -> xr.DataArray:
    """
    Standardize calendar for daily series.

    - mode='drop': drop Feb 29 → all years have 365 days
    - mode='keep': do nothing
    """
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
    elif mode == "keep":
        return da
    else:
        raise ValueError("FEB29_MODE must be 'drop' or 'keep'.")


def find_first_event(ts: xr.DataArray,
                     threshold: float,
                     k: int,
                     above: bool) -> int | float:
    """
    Return integer index (within ts.time) of the first k-day run meeting
    the threshold condition, or np.nan if none exists.

    - ts: 1D time-series DataArray (time)
    - threshold: SIC threshold (0–1)
    - k: persistence window length (days)
    - above=True: require ts >= threshold
    - above=False: require ts <= threshold

    Uses xarray rolling+construct to compute sliding 'all True' windows.
    """
    if above:
        cond = ts >= threshold
    else:
        cond = ts <= threshold

    # Rolling window over time dimension
    roll = cond.rolling(time=k).construct("window")  # (time, window)
    hits = roll.all("window")                        # True where k-day run exists, at window end

    if not bool(hits.any()):
        return np.nan

    # argmax over booleans returns first True because False=0, True=1
    return int(hits.argmax("time").item())


def slice_season(da: xr.DataArray, start_iso: str, end_iso: str) -> xr.DataArray:
    """
    Slice the DataArray by time between two ISO date strings (inclusive).

    Example:
        slice_season(ice365, "1980-02-15", "1980-09-30")
    """
    return da.sel(time=slice(start_iso, end_iso))


def save_year_field(out_dir: str,
                    year: int,
                    arr: np.ndarray,
                    varname: str,
                    template: xr.Dataset) -> None:
    """
    Save a single (y, x) field as a NetCDF with original x/y coords from template.

    - out_dir: directory to write into (will be created)
    - year: integer year
    - arr: 2D numpy array [ny, nx] of DOY or NaN
    - varname: "FS" or "MS"
    - template: Dataset with x,y coordinates to copy
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        {varname: (("y", "x"), arr)},
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
    Compute FS (advance) and MS (retreat) for one calendar year (y),
    using the standardized 365-day calendar DataArray (ice365).

    - ice365: SIC DataArray with calendar already standardized, dims (time, y, x)
    - year: integer year y (must have data for y and y+1)
    - threshold: SIC threshold (0–1)
    - k: persistence window length (days)
    - landmask: Boolean DataArray (y, x) where True indicates land/missing

    Returns:
        FS, MS: 2D numpy arrays [ny, nx] of DOY (1–365), NaN where undefined.
    """
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    y, y1 = year, year + 1

    # Retreat (MS): y-08-15 → (y+1)-02-28
    ts_MS = slice_season(ice365, f"{y}{MS_START_MMDD}", f"{y1}{MS_END_MMDD}")

    # Advance (FS): y-02-15 → y-09-30
    ts_FS = slice_season(ice365, f"{y}{FS_START_MMDD}", f"{y}{FS_END_MMDD}")

    # Safety: if the seasonal windows are pathologically short, just skip
    if ts_MS.time.size < max(60, k) and ts_FS.time.size < max(60, k):
        return FS, MS

    # Explicit loops: simple and readable; can vectorize later if needed
    for j in range(ny):
        # slice once along y to reduce indexing overhead
        col_MS = ts_MS.isel(y=j).transpose("time", "x")  # (t, x)
        col_FS = ts_FS.isel(y=j).transpose("time", "x")  # (t, x)

        for i in range(nx):
            # skip known land/missing immediately
            if bool(landmask.isel(y=j, x=i)):
                continue

            ts_r = col_MS[:, i]  # retreat series (Aug→Feb)
            ts_a = col_FS[:, i]  # advance series (Feb→Sep)

            # MS (retreat): first k-day run <= threshold
            if bool((ts_r <= threshold).any()):
                idx_rt = find_first_event(ts_r, threshold, k, above=False)
                if not np.isnan(idx_rt):
                    MS[j, i] = ts_r.time[int(idx_rt)].dt.dayofyear.item()

            # FS (advance): first k-day run >= threshold
            if bool((ts_a >= threshold).any()):
                idx_ad = find_first_event(ts_a, threshold, k, above=True)
                if not np.isnan(idx_ad):
                    FS[j, i] = ts_a.time[int(idx_ad)].dt.dayofyear.item()

    # Apply land mask explicitly
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
                        feb29_mode: str = FEB29_MODE) -> None:
    """
    Wrapper: runs FS and MS for a given (threshold, k) over a list/range of years.

    Writes two directory trees:

      - {output_root}/FS_thr{XX}_k{K}/FS_YYYY.nc
      - {output_root}/MS_thr{XX}_k{K}/MS_YYYY.nc
    """
    # Load SIC
    ds = xr.open_dataset(input_file)[[conc_var, "x", "y", "time"]]
    ice = ds[conc_var].astype("float32")

    # Scale if needed (if your SIC is 0..100 instead of 0..1)
    if float(ice.max()) > 1.5:
        ice = ice / 100.0

    # Mask land/missing (assumes >1 is invalid in native units; adjust if needed)
    ice = ice.where(ice < 1.1)

    # Standardize calendar
    ice365 = standardize_calendar(ice, mode=feb29_mode)

    # Permanent land mask: cells that are NA all year
    landmask = ice365.isnull().all("time")

    # Output directories
    dir_FS = os.path.join(output_root, f"FS_thr{int(thr * 100):02d}_k{k}")
    dir_MS = os.path.join(output_root, f"MS_thr{int(thr * 100):02d}_k{k}")
    Path(dir_FS).mkdir(parents=True, exist_ok=True)
    Path(dir_MS).mkdir(parents=True, exist_ok=True)

    # Determine years available after standardization
    all_years = np.unique(ice365.time.dt.year.values)
    years_run = [y for y in years if (y in all_years) and ((y + 1) in all_years)]

    for year in tqdm(years_run,
                     desc=f"Running {sensor_tag} thr={thr:.2f}, k={k}"):
        FS, MS = compute_FS_MS_for_year(ice365, year, thr, k, landmask)
        # Save one variable per file (keeps diffs/versioning simple)
        save_year_field(dir_FS, year, FS, "FS", ds)
        save_year_field(dir_MS, year, MS, "MS", ds)
        gc.collect()


# =======================
# CONVENIENCE DRIVERS
# =======================

def run_threshold_sweep(years,
                        k: int = 5,
                        thresholds: tuple[float, ...] = (0.10, 0.15, 0.30)) -> None:
    """
    Run FS/MS detection over a fixed k for several thresholds.
    """
    for thr in thresholds:
        run_phase_detection(thr=thr, k=k, years=years)


def run_window_sweep(years,
                     thr: float = 0.15,
                     windows: tuple[int, ...] = (3, 5, 7)) -> None:
    """
    Run FS/MS detection over a fixed threshold for several persistence windows k.
    """
    for k in windows:
        run_phase_detection(thr=thr, k=k, years=years)


# =======================
# EXAMPLE MAIN
# =======================

if __name__ == "__main__":
    # Define the span you actually have
    YEARS = range(1979, 2025)  # adjust as needed

    # A) Threshold sensitivity at fixed k=5
    run_threshold_sweep(years=YEARS, k=5, thresholds=(0.10, 0.15, 0.30))

    # B) Window sensitivity at fixed thr=0.15
    run_window_sweep(years=YEARS, thr=0.15, windows=(3, 5, 7))

    print("Done.")
