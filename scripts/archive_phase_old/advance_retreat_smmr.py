# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMMR Phase Detection (Static Method) with Wrapper
- FS (advance): first k-day run SIC >= threshold (Feb 15 → Sep 30)
- MS (retreat): first k-day run SIC <= threshold (Aug 15 → Feb 28 of next year)
- Persistence: rolling "all True" over k days
- Calendar: standardized to 365 DOY by default (drops Feb 29)
- Land/missing masking applied once, reused every year
- Wrapper to sweep thresholds and windows
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
THRESHOLD = [0.10, 0.15, 0.30]   # SMMR is 0-1
WINDOWS = [3, 5, 7]
SENSOR_TAG = "SMMR"

# Search windows (indices will be built per-year on a 365-day timeline)
# Southern Hemisphere:
#   MS (retreat): mid-Aug → end-Feb (cross-year)
#   FS (advance): mid-Feb → end-Sep
MS_START_MMDD = "-08-15"   # retreat search begins Aug 15 (year y)
MS_END_MMDD = "-02-28"   # ends Feb 28 (year y+1)
FS_START_MMDD = "-02-15"   # advance search begins Feb 15 (year y)
FS_END_MMDD = "-09-30"   # ends Sep 30 (year y)

# Feb 29 policy for standardization:
#   "drop": remove Feb 29 → every year has 365 days (recommended)
#   "keep": keep Feb 29 → some later comparisons need care
FEB29_MODE = "drop"

# === HELPER FUNCTION === #
def standardize_calendar(da: xr.DataArray, mode: str = "drop") -> xr.DataArray:
    """
    Standardize calendar for daily series.
    - mode='drop': drop Feb 29
    - mode='keep': do nothing (you'll need careful wrap logic later)
    """
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
    elif mode == "keep":
        return da
    else:
        raise ValueError("FEB29_MODE must be 'drop' or 'keep'.")

def find_first_event(ts: xr.DataArray, threshold: float, k: int, above: bool) -> int | float:
    """
    Return integer index (within ts.time) of first k-day run meeting condition, else np.nan.
    Uses xarray's rolling+construct to compute a sliding 'all True' test.
    """
    cond = (ts > threshold) if above else (ts < threshold)
    roll = cond.rolling(time=k).construct("window")
    hits = roll.all("window")
    if not bool(hits.any()):
        return np.nan
    # argmax returns the first True index because booleans are False(0)/True(1)
    return int(hits.argmax("time").item())

def slice_season(da: xr.DataArray, start_iso: str, end_iso: str) -> xr.DataArray:
    """
    Slice the DataArray by time between two ISO strings (inclusive).
    """
    return da.sel(time=slice(start_iso, end_iso))

def save_year_field(out_dir: str, year: int, arr: np.ndarray, varname: str, template: xr.Dataset):
    """
    Save a single (y,x) field as a NetCDF with original x/y coords from template.
    """
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
    Compute FS (advance) and MS (retreat) for one calendar year (y),
    using standardized 365-day calendar DataArray (ice365).

    Returns two (ny, nx) arrays of DOY (1..365) with NaN where undefined.
    """
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)

    # --- define seasonal windows (strings keep leap logic simple after standardization) ---
    y, y1 = year, year + 1
    # Retreat (MS): y-08-15 → (y+1)-02-28
    ts_MS = slice_season(ice365, f"{y}{MS_START_MMDD}", f"{y1}{MS_END_MMDD}")
    # Advance (FS): y-02-15 → y-09-30
    ts_FS = slice_season(ice365, f"{y}{FS_START_MMDD}", f"{y}{FS_END_MMDD}")

    # Safety: skip tiny windows (edge years)
    if ts_MS.time.size < max(60, k) and ts_FS.time.size < max(60, k):
        return FS, MS

    # --- iterate grid (explicit for clarity; can vectorize later) ---
    for j in range(ny):
        # slice once along y to reduce indexing overhead
        col_MS = ts_MS.isel(y=j).transpose("time", "x")   # (t, x)
        col_FS = ts_FS.isel(y=j).transpose("time", "x")   # (t, x)
        for i in range(nx):
            if landmask[j, i]:
                continue

            ts_r = col_MS[:, i]   # retreat series (Aug→Feb)
            ts_a = col_FS[:, i]   # advance series (Feb→Sep)

            # RETREAT (MS): first k-day run <= threshold
            if bool((ts_r < threshold).any()):
                idx_rt = find_first_event(ts_r, threshold, k, above=False)
                if not np.isnan(idx_rt):
                    MS[j, i] = ts_r.time[int(idx_rt)].dt.dayofyear.item()

            # ADVANCE (FS): first k-day run >= threshold
            if bool((ts_a > threshold).any()):
                idx_ad = find_first_event(ts_a, threshold, k, above=True)
                if not np.isnan(idx_ad):
                    FS[j, i] = ts_a.time[int(idx_ad)].dt.dayofyear.item()

    # apply land mask explicitly
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
      - {output_root}/FS_thr{XX}_k{K}/year.nc
      - {output_root}/MS_thr{XX}_k{K}/year.nc
    """
    ds = xr.open_dataset(input_file)[[conc_var, "x", "y", "time"]]
    ice = ds[conc_var].astype("float32")

    # Scale if needed (if your SIC is 0..100 instead of 0..1):
    if float(ice.max()) > 1.5:
        ice = ice / 100.0

    # mask land/missing (assumes >1 is invalid in native units—adjust if needed)
    ice = ice.where(ice < 1.1)

    # standardize calendar (drop Feb 29 by default)
    ice365 = standardize_calendar(ice, mode=feb29_mode)

    # permanent land mask: cells that are NA all year (and all years if you prefer)
    landmask = ice365.isnull().all("time")

    # output dirs
    dir_FS = os.path.join(output_root, f"FS_thr{int(thr*100):02d}_k{k}")
    dir_MS = os.path.join(output_root, f"MS_thr{int(thr*100):02d}_k{k}")
    Path(dir_FS).mkdir(parents=True, exist_ok=True)
    Path(dir_MS).mkdir(parents=True, exist_ok=True)

    # determine years available after standardization
    all_years = np.unique(ice365.time.dt.year.values)
    years_run = [y for y in years if y in all_years and (y+1) in all_years]  # need y and y+1 for MS window

    for year in tqdm(years_run, desc=f"Running {sensor_tag} thr={thr:.2f}, k={k}"):
        FS, MS = compute_FS_MS_for_year(ice365, year, thr, k, landmask)
        # Save one var per file (keeps diffs very easy)
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
    # Define the SMMR span you actually have
    YEARS = range(1979, 2025)  # edit for your dataset

    # A) Threshold sensitivity at fixed k=5
    run_threshold_sweep(years=YEARS, k=5, thresholds=(0.10, 0.15, 0.30))

    # B) Window sensitivity at fixed thr=0.15
    run_window_sweep(years=YEARS, thr=0.15, windows=(3, 5, 7))

    print("Done.")

#changes