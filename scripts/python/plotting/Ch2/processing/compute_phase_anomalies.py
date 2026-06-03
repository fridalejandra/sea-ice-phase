#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_phase_anomalies.py

Computes climatologies and anomalies for FS and MS phase dates,
for both static and dynamic methods, for both SMMR and AMSR-E.

Expects input structure under data/:
    data/
        SMMR_phase/
            static/thr15_k5/FS/FS_YYYY.nc
            static/thr15_k5/MS/MS_YYYY.nc
            dynamic/k5_q70/FS/FS_YYYY.nc
            dynamic/k5_q70/MS/MS_YYYY.nc
        AMSRE_phase/
            static/thr15_k5/FS/FS_YYYY.nc
            ...

Outputs under data/anomalies/{sensor}/:
    FS_static_thr15_k5_climatology.nc
    FS_static_thr15_k5_anomalies.nc
    MS_static_thr15_k5_climatology.nc
    MS_static_thr15_k5_anomalies.nc
    FS_dynamic_k5_q70_climatology.nc
    ...

Usage:
    python compute_phase_anomalies.py                        # SMMR baseline only
    python compute_phase_anomalies.py --sensor AMSRE
    python compute_phase_anomalies.py --sensor all
    python compute_phase_anomalies.py --thr 15 20 30         # multiple thresholds
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase/data")

SENSOR_YEARS = {
    "SMMR":  (1979, 2025),
    "AMSRE": (2012, 2025),
}

# Static parameter combinations to process
THRESHOLDS = [15, 20, 30]   # integer percent e.g. 15 = thr15
WINDOWS    = [3, 5, 7]
BASELINE_THR = 15
BASELINE_K   = 5

# Dynamic settings
DYN_K  = 5
DYN_Q  = 70   # 70th percentile

# MS wrap anchor — Aug 15
AUG15_DOY = 227

# =============================================================================
# HELPERS
# =============================================================================

def ms_to_days_since_aug15(da: xr.DataArray) -> xr.DataArray:
    """Convert MS calendar DOY to continuous days-since-Aug-15."""
    return xr.where(da < AUG15_DOY, da + 365, da) - AUG15_DOY


def load_phase_year(path: Path, varname: str, year: int) -> xr.DataArray | None:
    fpath = path / f"{varname}_{year}.nc"
    if not fpath.exists():
        return None
    ds = xr.open_dataset(fpath)
    if varname not in ds:
        ds.close()
        return None
    da = ds[varname].load()
    ds.close()
    if not np.any(np.isfinite(da.values)):
        return None
    return da.transpose("y", "x")


def build_series(path: Path, varname: str,
                 year_start: int, year_end: int) -> xr.DataArray | None:
    """Stack annual arrays into a (year, y, x) DataArray."""
    arrays, years = [], []
    for y in range(year_start, year_end + 1):
        da = load_phase_year(path, varname, y)
        if da is not None:
            arrays.append(da.expand_dims(year=[y]))
            years.append(y)
    if not arrays:
        return None
    return xr.concat(arrays, dim="year").assign_coords(year=("year", years))


def compute_clim_anom(series: xr.DataArray, phase: str, tag: str,
                      out_dir: Path) -> None:
    """Compute and save climatology + anomalies. For MS also compute dsa version."""
    out_dir.mkdir(parents=True, exist_ok=True)

    clim = series.mean("year", skipna=True)
    anom = series - clim

    clim_name = f"{phase}_{tag}_clim"
    anom_name = f"{phase}_{tag}_anom"

    clim_ds = clim.to_dataset(name=clim_name)
    anom_ds = anom.to_dataset(name=anom_name)

    # MS: also compute days-since-Aug-15 version
    if phase == "MS":
        series_dsa = ms_to_days_since_aug15(series)
        clim_dsa   = series_dsa.mean("year", skipna=True)
        anom_dsa   = series_dsa - clim_dsa

        clim_ds[f"{clim_name}_dsa"] = clim_dsa
        clim_ds[f"{clim_name}_dsa"].attrs = {
            "units": "days since Aug 15",
            "description": "MS in continuous seasonal coordinate, avoids year-boundary wrap"
        }
        anom_ds[f"{anom_name}_dsa"] = anom_dsa
        anom_ds[f"{anom_name}_dsa"].attrs = {
            "units": "days",
            "description": "MS anomalies in days-since-Aug-15 space"
        }

    clim_path = out_dir / f"{phase}_{tag}_climatology.nc"
    anom_path = out_dir / f"{phase}_{tag}_anomalies.nc"

    print(f"  saving {clim_path.name}")
    clim_ds.to_netcdf(clim_path)
    print(f"  saving {anom_path.name}")
    anom_ds.to_netcdf(anom_path)


# =============================================================================
# RUNNERS
# =============================================================================

def run_static(sensor: str, thresholds: list[int], windows: list[int]) -> None:
    year_start, year_end = SENSOR_YEARS[sensor]
    base = DATA_ROOT / f"{sensor}_phase" / "static"
    out_base = DATA_ROOT / "anomalies" / sensor

    for thr in thresholds:
        for k in windows:
            tag = f"static_thr{thr:02d}_k{k}"
            print(f"\n=== {sensor} {tag} ===")
            for phase in ["FS", "MS"]:
                path = base / f"thr{thr:02d}_k{k}" / phase
                series = build_series(path, phase, year_start, year_end)
                if series is None:
                    print(f"  no data found at {path} — skipping")
                    continue
                compute_clim_anom(series, phase, tag, out_base)


def run_dynamic(sensor: str) -> None:
    year_start, year_end = SENSOR_YEARS[sensor]
    base = DATA_ROOT / f"{sensor}_phase" / "dynamic" / f"k{DYN_K}_q{DYN_Q}"
    out_base = DATA_ROOT / "anomalies" / sensor
    tag = f"dynamic_k{DYN_K}_q{DYN_Q}"

    print(f"\n=== {sensor} {tag} ===")
    for phase in ["FS", "MS"]:
        path = base / phase
        series = build_series(path, phase, year_start, year_end)
        if series is None:
            print(f"  no data found at {path} — skipping")
            continue
        compute_clim_anom(series, phase, tag, out_base)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compute FS/MS climatologies and anomalies.")
    p.add_argument("--sensor", default="SMMR", choices=["SMMR", "AMSRE", "all"])
    p.add_argument("--thr", type=int, nargs="+", default=THRESHOLDS)
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    p.add_argument("--baseline-only", action="store_true",
                   help=f"Baseline only: thr={BASELINE_THR} k={BASELINE_K}")
    p.add_argument("--method", default="all", choices=["static", "dynamic", "all"])
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    sensors = ["SMMR", "AMSRE"] if args.sensor == "all" else [args.sensor]
    thrs    = [BASELINE_THR] if args.baseline_only else args.thr
    wins    = [BASELINE_K]   if args.baseline_only else args.windows
    methods = ["static", "dynamic"] if args.method == "all" else [args.method]

    for sensor in sensors:
        for method in methods:
            if method == "static":
                run_static(sensor, thrs, wins)
            else:
                run_dynamic(sensor)

    print("\nDone.")