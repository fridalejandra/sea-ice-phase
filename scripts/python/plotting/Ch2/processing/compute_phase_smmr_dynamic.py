#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run dynamic-threshold phase dates using a percentile + slope detector.
- You choose which dynamic schemes and parameters to test.
- k fixed to 5 by default (fast).
- Reuses baseline stats (μ, σ, Q) per year to avoid recomputation.
- Outputs FS/MS/ME per year as NetCDF, like your static runs.

To keep runtime reasonable:
  * Start with YEARS_SAMPLE (few years) to sanity-check.
  * Then flip USE_FULL_YEARS=True.
"""

import os, gc
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
SENSOR       = "SMMR"
INPUT_FILE   = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
CONC_VAR     = "N07_ICECON"

# Output root (local). You can rclone later in one go.
OUT_ROOT     = f"/user/geog/falejandraperez/sea-ice-phase/results/dynamic_thresholds/"

# Years
FULL_YEARS   = range(1979, 2025)
#YEARS_SAMPLE = [1984, 1992, 2007, 2016, 2023]  # tweak to taste
USE_FULL_YEARS = True  # set True after quick check

# Detector settings
K            = 5               # persistence window (keep at 5 for speed)
FEB29_MODE   = "drop"          # standardize calendar to 365

# Search windows (match your static script)
MS_START_MMDD = "-08-15"; MS_END_MMDD = "-02-28"
FS_START_MMDD = "-02-15"; FS_END_MMDD = "-09-30"

# Dynamic schemes to run (dynamic percentile + slope condition)
RUN_SCHEMES = [
    ("quantile_slope", {
        "p": 0.70,                     # SIC percentile
        "dC_min": 0.03,                # minimum daily SIC change (0-1 units)
        "win_fs": ("02-01", "03-31"),  # late-winter window for FS threshold
        "win_ms": ("08-01", "09-30"),  # late-summer window for MS threshold
    }),
]


# -------------- helpers -------------- #
def standardize_calendar(da: xr.DataArray, mode="drop"):
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month==2) & (da.time.dt.day==29)))
    if mode == "keep":
        return da
    raise ValueError

def slice_season(da, start_iso, end_iso):
    return da.sel(time=slice(start_iso, end_iso))

def find_first_event_dyn(ts, thr_scalar, k, above: bool):
    """Return index of first k-day run meeting condition, else nan."""
    cond = (ts > thr_scalar) if above else (ts < thr_scalar)
    hits = cond.rolling(time=k).construct("window").all("window")
    if not bool(hits.any()):
        return np.nan
    return int(hits.argmax("time").item())

def find_last_event_dyn(ts, thr_scalar, k, above: bool):
    """
    Return index of LAST k-day run meeting condition, else nan.
    Mirror of find_first_event_dyn but from the end.
    """
    cond = (ts > thr_scalar) if above else (ts < thr_scalar)
    hits = cond.rolling(time=k).construct("window").all("window")
    if not bool(hits.any()):
        return np.nan
    idxs = np.where(hits.values)[0]
    if idxs.size == 0:
        return np.nan
    return int(idxs[-1])


def first_run_k_bool(cond: np.ndarray, k: int):
    """Index of first k-day run of True in 1D boolean array, else nan."""
    n = cond.size
    if n < k or not cond.any():
        return np.nan
    for t in range(0, n - k + 1):
        if cond[t] and cond[t:t+k].all():
            return t
    return np.nan

def last_run_k_bool(cond: np.ndarray, k: int):
    """Index of last k-day run of True in 1D boolean array, else nan."""
    n = cond.size
    if n < k or not cond.any():
        return np.nan
    for t in range(n - k, -1, -1):
        if cond[t] and cond[t:t+k].all():
            return t
    return np.nan


def save_year_field(out_dir, year, arr, varname, template_ds):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"{varname}_{year}.nc")
    ny, nx = arr.shape
    da_out = xr.DataArray(
        arr,
        coords={"y": template_ds["y"], "x": template_ds["x"]},
        dims=("y", "x"),
        name=varname
    )
    da_out.attrs["long_name"] = f"{varname} day-of-year"
    da_out.attrs["sensor"] = SENSOR
    ds_out = da_out.to_dataset()
    ds_out.to_netcdf(out_path, mode="w")
    print(f"  wrote {out_path}")


# ---------- threshold builders ---------- #
def _late_window(ice365, year, win):
    """
    Extract a simple month-day window in a given year.

    win: (start_mm_dd, end_mm_dd) e.g. ("02-01", "03-31")
    """
    start_iso = f"{year}-{win[0]}"
    end_iso   = f"{year}-{win[1]}"
    da = ice365.sel(time=slice(start_iso, end_iso))
    return da

def thr_mu_sigma(ice, year, phase, params):
    """μ ± α σ thresholds as a reference scheme."""
    if phase == "FS":
        da = _late_window(ice, year, params["win_fs"])
        mu = da.mean("time", skipna=True)
        sig = da.std("time", skipna=True)
        T = mu + params["alpha"]*sig
        T = xr.where(T < 0.15, 0.15, T)
        return T.astype("float32")
    else:
        da = _late_window(ice, year, params["win_ms"])
        mu = da.mean("time", skipna=True)
        sig = da.std("time", skipna=True)
        T = mu - params["alpha"]*sig
        T = xr.where(T > 0.60, 0.60, T)
        return T.astype("float32")

def thr_quantile(ice, year, phase, params):
    """Percentile thresholds in late winter / late summer."""
    if phase == "FS":
        da = _late_window(ice, year, params["win_fs"])
        T = da.quantile(params["p"], dim="time", skipna=True)
        T = xr.where(T < 0.15, 0.15, T)
        return T.astype("float32")
    else:
        da = _late_window(ice, year, params["win_ms"])
        T = da.quantile(1.0-params["p"], dim="time", skipna=True)
        T = xr.where(T > 0.60, 0.60, T)
        return T.astype("float32")

def thr_slope_scaled(ice, year, phase, params):
    # simple slope proxy: mean positive (FS) / negative (MS) daily diff magnitude
    if phase == "FS":
        da = _late_window(ice, year, params["win_fs"])
        H = (da.diff("time").clip(min=0)).mean("time", skipna=True)
        T = 0.15 + params["gamma"] * H
        T = xr.where(T < 0.10, 0.10, xr.where(T > 0.30, 0.30, T))
        return T.astype("float32")
    else:
        da = _late_window(ice, year, params["win_ms"])
        H = (-da.diff("time").clip(min=0)).mean("time", skipna=True)
        T = 0.60 - params["gamma"] * H
        T = xr.where(T < 0.50, 0.50, xr.where(T > 0.70, 0.70, T))
        return T.astype("float32")

THR_BUILDERS = {
    "mu_sigma":       thr_mu_sigma,
    "quantile":       thr_quantile,
    "slope_scaled":   thr_slope_scaled,
    "quantile_slope": thr_quantile,  # same SIC threshold, extra slope logic in detector
}


# ---------- per-year compute using dynamic T(y,x) ---------- #
def compute_FS_MS_ME_year_dyn(ice365, year, T_FS, T_MS, k, landmask, template_ds,
                              scheme, params):
    ny, nx = ice365.y.size, ice365.x.size
    FS = np.full((ny, nx), np.nan, dtype=float)
    MS = np.full((ny, nx), np.nan, dtype=float)
    ME = np.full((ny, nx), np.nan, dtype=float)

    y, y1 = year, year + 1
    ts_MS = slice_season(ice365, f"{y}{MS_START_MMDD}", f"{y1}{MS_END_MMDD}")
    ts_FS = slice_season(ice365, f"{y}{FS_START_MMDD}", f"{y}{FS_END_MMDD}")

    if ts_MS.time.size < max(60, k) and ts_FS.time.size < max(60, k):
        return FS, MS, ME

    for j in range(ny):
        col_MS = ts_MS.isel(y=j).transpose("time", "x")
        col_FS = ts_FS.isel(y=j).transpose("time", "x")
        for i in range(nx):
            if landmask[j, i]:
                continue

            thr_ms = float(T_MS.values[j, i])
            thr_fs = float(T_FS.values[j, i])
            ts_r = col_MS[:, i]
            ts_a = col_FS[:, i]

            # --------------- quantile_slope: SIC percentile + ΔSIC --------------- #
            if scheme == "quantile_slope":
                dC_min = float(params.get("dC_min", 0.03))

                # retreat / melt start (MS): need k days BELOW threshold
                # AND k days of sufficiently negative slope
                if np.isfinite(thr_ms) and bool((ts_r < thr_ms).any()):
                    arr_r = ts_r.values.astype(float)
                    if arr_r.size > 1:
                        dC_r = np.diff(arr_r)
                        conc_cond_ms = arr_r[1:] < thr_ms
                        slope_cond_ms = dC_r <= -dC_min
                        cond_ms = conc_cond_ms & slope_cond_ms
                        idx0 = first_run_k_bool(cond_ms, k)
                        if not np.isnan(idx0):
                            MS[j, i] = ts_r.time[int(idx0) + 1].dt.dayofyear.item()

                # ME: keep as percentile-only (no slope condition) for now
                if np.isfinite(thr_ms) and bool((ts_r < thr_ms).any()):
                    idx_last = find_last_event_dyn(ts_r, thr_ms, k, above=False)
                    if not np.isnan(idx_last):
                        ME[j, i] = ts_r.time[int(idx_last)].dt.dayofyear.item()

                # advance / freeze start (FS): k days ABOVE threshold
                # AND k days of sufficiently positive slope
                if np.isfinite(thr_fs) and bool((ts_a > thr_fs).any()):
                    arr_a = ts_a.values.astype(float)
                    if arr_a.size > 1:
                        dC_a = np.diff(arr_a)
                        conc_cond_fs = arr_a[1:] > thr_fs
                        slope_cond_fs = dC_a >= dC_min
                        cond_fs = conc_cond_fs & slope_cond_fs
                        idx_fs = first_run_k_bool(cond_fs, k)
                        if not np.isnan(idx_fs):
                            FS[j, i] = ts_a.time[int(idx_fs) + 1].dt.dayofyear.item()

                # move on to next grid cell
                continue

            # --------------- default: percentile/mu_sigma/slope_scaled only --------------- #
            # MS: first k-day run BELOW threshold
            if np.isfinite(thr_ms) and bool((ts_r < thr_ms).any()):
                idx = find_first_event_dyn(ts_r, thr_ms, k, above=False)
                if not np.isnan(idx):
                    MS[j, i] = ts_r.time[int(idx)].dt.dayofyear.item()

            # ME: last k-day run BELOW threshold
            if np.isfinite(thr_ms) and bool((ts_r < thr_ms).any()):
                idx_last = find_last_event_dyn(ts_r, thr_ms, k, above=False)
                if not np.isnan(idx_last):
                    ME[j, i] = ts_r.time[int(idx_last)].dt.dayofyear.item()

            # FS: first k-day run ABOVE threshold
            if np.isfinite(thr_fs) and bool((ts_a > thr_fs).any()):
                idx = find_first_event_dyn(ts_a, thr_fs, k, above=True)
                if not np.isnan(idx):
                    FS[j, i] = ts_a.time[int(idx)].dt.dayofyear.item()

    FS[landmask.values] = np.nan
    MS[landmask.values] = np.nan
    ME[landmask.values] = np.nan
    return FS, MS, ME


# ---------------- main runner ---------------- #
def main():
    years = list(FULL_YEARS if USE_FULL_YEARS else YEARS_SAMPLE)

    ds = xr.open_dataset(INPUT_FILE)[[CONC_VAR, "x", "y", "time"]]
    ice = ds[CONC_VAR].astype("float32")
    if float(ice.max()) > 1.5:
        ice = ice/100.0
    ice = ice.where(ice < 1.1)  # mask invalid
    ice365 = standardize_calendar(ice, FEB29_MODE)
    landmask = ice365.isnull().all("time")

    all_years = np.unique(ice365.time.dt.year.values)
    run_years = [y for y in years if y in all_years and (y+1) in all_years]

    for scheme, params in RUN_SCHEMES:
        builder = THR_BUILDERS[scheme]
        out_dir_FS = os.path.join(OUT_ROOT, f"{scheme}_k{K}", "FS")
        out_dir_MS = os.path.join(OUT_ROOT, f"{scheme}_k{K}", "MS")
        out_dir_ME = os.path.join(OUT_ROOT, f"{scheme}_k{K}", "ME")
        Path(out_dir_FS).mkdir(parents=True, exist_ok=True)
        Path(out_dir_MS).mkdir(parents=True, exist_ok=True)
        Path(out_dir_ME).mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running scheme: {scheme} | params={params} | k={K} ===")
        for year in tqdm(run_years, desc=f"{scheme}"):
            # Build per-pixel threshold fields once per year
            T_FS = builder(ice365, year, "FS", params)
            T_MS = builder(ice365, year, "MS", params)

            FS, MS, ME = compute_FS_MS_ME_year_dyn(
                ice365, year, T_FS, T_MS, K, landmask, ds, scheme, params
            )

            tag = "_".join(
                [f"{k}{v}" for k, v in params.items()
                 if k in ("alpha", "p", "gamma", "dC_min")]
            )
            save_year_field(os.path.join(out_dir_FS, tag), year, FS, "FS", ds)
            save_year_field(os.path.join(out_dir_MS, tag), year, MS, "MS", ds)
            save_year_field(os.path.join(out_dir_ME, tag), year, ME, "ME", ds)

            gc.collect()

    print("Done.")

if __name__ == "__main__":
    main()
