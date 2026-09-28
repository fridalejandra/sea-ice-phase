#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_transition_metrics.py

Computes three transition structure metrics for FS and MS:

  1. Crossing frequency  — number of threshold crossings during the search window
  2. Transition duration — DOY span from first to last crossing before final sustained run
  3. Method-spread ambiguity — std of phase dates across all static threshold/window combos

Outputs under data/transition_metrics/{sensor}/:

  crossing_freq_FS_thr15.nc    (year, y, x)
  crossing_freq_MS_thr15.nc    (year, y, x)
  transition_dur_FS_thr15.nc   (year, y, x)
  transition_dur_MS_thr15.nc   (year, y, x)
  method_spread_FS.nc          (year, y, x)  -- across all thr/k combos
  method_spread_MS.nc          (year, y, x)

Usage:
  python compute_transition_metrics.py                   # SMMR all thresholds
  python compute_transition_metrics.py --sensor AMSRE
  python compute_transition_metrics.py --sensor all
  python compute_transition_metrics.py --no-ambiguity    # skip method spread
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase/data")
OUT_ROOT  = DATA_ROOT / "transition_metrics"

SENSOR_CONFIGS = {
    "SMMR": {
        "input_file": DATA_ROOT / "merged" / "SMMR_merged_19781101_20251231_complete.nc",
        "conc_var":   "N07_ICECON",
        "units":      "fraction",
        "mask_above": 1.1,
        "years":      range(1979, 2026),
        "phase_root": DATA_ROOT / "SMMR_phase",
    },
    "AMSRE": {
        "input_file": DATA_ROOT / "merged" / "AMSRE_merged_07132012_08312025.nc",
        "conc_var":   "SI_12km_SH_ICECON_DAY_SpPolarGrid12km",
        "units":      "percent",
        "mask_above": 110,
        "years":      range(2012, 2026),
        "phase_root": DATA_ROOT / "AMSRE_phase",
    },
}

THRESHOLDS   = [0.15, 0.20, 0.30]
WINDOWS      = [3, 5, 7]
BASELINE_K   = 5

MS_START_MMDD = "-08-15"
MS_END_MMDD   = "-02-28"
FS_START_MMDD = "-02-15"
FS_END_MMDD   = "-09-30"


# =============================================================================
# HELPERS
# =============================================================================

def load_sic(sensor: str) -> xr.DataArray:
    cfg = SENSOR_CONFIGS[sensor]
    ds  = xr.open_dataset(cfg["input_file"])
    ice = ds[cfg["conc_var"]].astype("float32")
    ice = ice.where(ice <= cfg["mask_above"])
    if cfg["units"] == "percent":
        ice = ice / 100.0
    # drop Feb 29, sort
    ice = ice.sel(time=~((ice.time.dt.month == 2) & (ice.time.dt.day == 29)))
    return ice.sortby("time")


def slice_season(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    return da.sel(time=slice(start, end))


# =============================================================================
# METRIC 1: CROSSING FREQUENCY (vectorized)
# =============================================================================

def crossing_frequency_vectorized(ts: xr.DataArray,
                                   threshold: float,
                                   above_for_event: bool) -> np.ndarray:
    """
    Count threshold crossings during the seasonal window.

    A crossing is defined as a sign change in (SIC - threshold).
    For FS (above=True): count times SIC crosses threshold upward OR downward.
    For MS (above=False): same — any crossing counts as flickering.

    Returns (y, x) array of crossing counts.
    """
    diff = (ts - threshold).values   # (time, y, x)

    # sign: +1 above threshold, -1 below, 0 exactly on
    sign = np.sign(diff)

    # crossing = sign change between consecutive timesteps
    # ignore zeros (exactly on threshold)
    sign_nonzero = np.where(sign == 0, np.nan, sign)

    # forward fill NaNs to handle exact-threshold cases
    # (use simple loop over time for robustness)
    for t in range(1, sign_nonzero.shape[0]):
        mask = np.isnan(sign_nonzero[t])
        sign_nonzero[t] = np.where(mask, sign_nonzero[t-1], sign_nonzero[t])

    # count sign changes
    changes = np.diff(sign_nonzero, axis=0)   # (time-1, y, x)
    crossings = np.nansum(np.abs(changes) > 0, axis=0).astype(float)  # (y, x)

    # mask cells with all-NaN
    all_nan = np.all(np.isnan(diff), axis=0)
    crossings[all_nan] = np.nan

    return crossings


# =============================================================================
# METRIC 2: TRANSITION DURATION (vectorized)
# =============================================================================

def transition_duration_vectorized(ts: xr.DataArray,
                                    threshold: float,
                                    above_for_event: bool,
                                    k: int) -> np.ndarray:
    """
    Compute transition duration: DOY of last crossing before final sustained
    run minus DOY of first crossing.

    Short duration = sharp transition.
    Long duration  = diffuse/flickering transition.
    NaN where fewer than 2 crossings or no sustained run found.

    Returns (y, x) array of durations in days.
    """
    vals  = ts.values        # (time, y, x)
    doys  = ts.time.dt.dayofyear.values  # (time,)
    nt    = vals.shape[0]
    ny, nx = vals.shape[1], vals.shape[2]

    if above_for_event:
        cond = vals >= threshold
    else:
        cond = vals <= threshold

    # find first and last crossing indices vectorized
    # first crossing: first True in cond
    # last crossing before sustained run: last True before k consecutive True

    duration = np.full((ny, nx), np.nan, dtype=float)

    # sustained run mask (trailing rolling all-True)
    # compute via cumsum trick for speed
    # pad to handle edges
    cond_int = cond.astype(np.float32)

    # for each cell find first crossing and last crossing before sustained run
    # vectorize over y,x by reshaping to (time, ny*nx)
    c2d = cond_int.reshape(nt, -1)   # (time, ny*nx)

    n_cells = c2d.shape[1]
    first_cross = np.full(n_cells, np.nan)
    last_cross  = np.full(n_cells, np.nan)

    for ci in range(n_cells):
        col = c2d[:, ci]
        if np.all(np.isnan(col)):
            continue

        cross_idx = np.where(col > 0)[0]
        if len(cross_idx) == 0:
            continue

        first_cross[ci] = doys[cross_idx[0]]

        # find last crossing before sustained k-day run
        # sustained run: k consecutive True
        sustained_start = None
        for t in range(nt - k + 1):
            if all(col[t:t+k] > 0):
                sustained_start = t
                break

        if sustained_start is None:
            continue

        # last crossing is the last True index before sustained_start
        pre_sustained = cross_idx[cross_idx < sustained_start]
        if len(pre_sustained) > 0:
            last_cross[ci] = doys[pre_sustained[-1]]
        else:
            last_cross[ci] = doys[sustained_start]

    dur_flat = last_cross - first_cross
    dur_flat[dur_flat < 0] = np.nan   # wrap artifact
    duration = dur_flat.reshape(ny, nx)

    return duration


# =============================================================================
# METRIC 3: METHOD-SPREAD AMBIGUITY
# =============================================================================

def compute_method_spread(sensor: str, phase: str,
                           years: list[int]) -> xr.DataArray | None:
    """
    Compute std of phase dates across all static thr/k combinations.
    Returns (year, y, x) DataArray.
    """
    cfg = SENSOR_CONFIGS[sensor]
    phase_root = cfg["phase_root"] / "static"

    all_arrays = []
    for thr in THRESHOLDS:
        for k in WINDOWS:
            tag = f"thr{int(thr*100):02d}_k{k}"
            path = phase_root / tag / phase
            year_arrays = []
            for y in years:
                fpath = path / f"{phase}_{y}.nc"
                if not fpath.exists():
                    continue
                ds = xr.open_dataset(fpath)
                if phase not in ds:
                    ds.close()
                    continue
                da = ds[phase].load().expand_dims(year=[y])
                ds.close()
                year_arrays.append(da)
            if year_arrays:
                all_arrays.append(xr.concat(year_arrays, dim="year"))

    if not all_arrays:
        return None

    # stack all combos and compute std across them
    stacked = xr.concat(all_arrays, dim="combo")
    spread  = stacked.std("combo", skipna=True)
    return spread   # (year, y, x)


# =============================================================================
# RUNNER
# =============================================================================

def run_sensor(sensor: str, compute_ambiguity: bool = True) -> None:
    cfg      = SENSOR_CONFIGS[sensor]
    out_dir  = OUT_ROOT / sensor
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {sensor}")
    print(f"{'='*60}")

    # load SIC once
    print("Loading SIC...")
    ice = load_sic(sensor)
    all_years = np.unique(ice.time.dt.year.values)
    years_run = [y for y in cfg["years"]
                 if (y in all_years) and ((y + 1) in all_years)]

    landmask = ice.isnull().all("time").values

    for thr in THRESHOLDS:
        thr_tag = f"thr{int(thr*100):02d}"
        print(f"\n  threshold = {thr:.2f}")

        for phase, start_mm, end_mm, above in [
            ("FS", FS_START_MMDD, FS_END_MMDD,   True),
            ("MS", MS_START_MMDD, MS_END_MMDD,   False),
        ]:
            freq_years = []
            dur_years  = []

            for year in tqdm(years_run, desc=f"  {phase} {thr_tag}"):
                y1 = year + 1 if phase == "MS" else year
                ts = slice_season(ice,
                                  f"{year}{start_mm}",
                                  f"{y1}{end_mm}")
                if ts.time.size < BASELINE_K:
                    freq_years.append(
                        xr.full_like(ice.isel(time=0), np.nan).expand_dims(year=[year]))
                    dur_years.append(
                        xr.full_like(ice.isel(time=0), np.nan).expand_dims(year=[year]))
                    continue

                freq = crossing_frequency_vectorized(ts, thr, above)
                dur  = transition_duration_vectorized(ts, thr, above, BASELINE_K)

                freq[landmask] = np.nan
                dur[landmask]  = np.nan

                freq_da = xr.DataArray(freq, dims=["y","x"],
                                       coords={"y": ice.y, "x": ice.x}
                                       ).expand_dims(year=[year])
                dur_da  = xr.DataArray(dur,  dims=["y","x"],
                                       coords={"y": ice.y, "x": ice.x}
                                       ).expand_dims(year=[year])

                freq_years.append(freq_da)
                dur_years.append(dur_da)

            # stack and save
            freq_stack = xr.concat(freq_years, dim="year")
            dur_stack  = xr.concat(dur_years,  dim="year")

            freq_path = out_dir / f"crossing_freq_{phase}_{thr_tag}.nc"
            dur_path  = out_dir / f"transition_dur_{phase}_{thr_tag}.nc"

            freq_stack.to_dataset(name="crossing_freq").to_netcdf(freq_path)
            dur_stack.to_dataset(name="transition_dur").to_netcdf(dur_path)
            print(f"    saved {freq_path.name}")
            print(f"    saved {dur_path.name}")

    # method spread ambiguity
    if compute_ambiguity:
        print("\n  computing method-spread ambiguity...")
        for phase in ["FS", "MS"]:
            spread = compute_method_spread(sensor, phase, list(years_run))
            if spread is not None:
                out_path = out_dir / f"method_spread_{phase}.nc"
                spread.to_dataset(name="method_spread").to_netcdf(out_path)
                print(f"    saved {out_path.name}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute transition structure metrics.")
    p.add_argument("--sensor", default="SMMR",
                   choices=["SMMR", "AMSRE", "all"])
    p.add_argument("--no-ambiguity", action="store_true",
                   help="Skip method-spread ambiguity computation")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    sensors = ["SMMR", "AMSRE"] if args.sensor == "all" else [args.sensor]

    for sensor in sensors:
        run_sensor(sensor, compute_ambiguity=not args.no_ambiguity)

    print("\nDone.")