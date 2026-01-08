#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute FS/MS climatologies and anomalies for static and dynamic methods.

Static:
  /results/SMMR_phase/seaice_phases_SMMR_YYYY.nc
    - advance_YYYY -> FS_static
    - retreat_YYYY -> MS_static

Dynamic:
  /results/static_v2_slopeH/dynamic/quantile_k5/FS/p0.7/FS_YYYY.nc
  /results/static_v2_slopeH/dynamic/quantile_k5/MS/p0.7/MS_YYYY.nc
    - FS or MS variable inside

Outputs (under results/anomalies/):

  FS_static_climatology.nc     (FS_static_clim[y,x])
  MS_static_climatology.nc     (MS_static_clim[y,x])  + MS_static_clim_dsa[y,x]
  FS_dynamic_climatology.nc
  MS_dynamic_climatology.nc    (MS_dynamic_clim[y,x]) + MS_dynamic_clim_dsa[y,x]

  FS_static_anomalies.nc       (FS_static_anom[year,y,x])
  MS_static_anomalies.nc       (MS_static_anom[year,y,x]) + MS_static_anom_dsa[year,y,x]
  FS_dynamic_anomalies.nc
  MS_dynamic_anomalies.nc      (MS_dynamic_anom[year,y,x]) + MS_dynamic_anom_dsa[year,y,x]

Notes:
- FS climatologies/anomalies are computed in calendar DOY space (fine).
- MS crosses the year boundary, so we additionally compute MS in a continuous
  "days since Aug 15" coordinate to avoid calendar wrap artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")

STATIC_DIR = PROJECT_ROOT / "results" / "SMMR_phase"
DYN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "static_v2_slopeH"
    / "dynamic"
    / "quantile_k5"
)
DYN_DIR_FS = DYN_ROOT / "FS" / "p0.7"
DYN_DIR_MS = DYN_ROOT / "MS" / "p0.7"

OUT_DIR = PROJECT_ROOT / "results" / "anomalies"

YEAR_START = 1980
YEAR_END = 2023

# MS wrap anchor
AUG15_DOY = 227  # Aug 15 (non-leap)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def ms_to_days_since_aug15(ms_da: xr.DataArray, aug15_doy: int = AUG15_DOY) -> xr.DataArray:
    """
    Convert MS calendar DOY to continuous 'days since Aug 15'.

    Aug 15 -> 0, Dec 31 -> ~138, Jan 1 -> ~139, Feb 28 -> ~197.

    Assumes ms_da contains calendar DOY values in [1, 366] with NaNs for no-event.
    """
    ms_wrapped = xr.where(ms_da < aug15_doy, ms_da + 365, ms_da)
    return ms_wrapped - aug15_doy


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------

def load_static_year(phase: str, year: int) -> xr.DataArray | None:
    fpath = STATIC_DIR / f"seaice_phases_SMMR_{year}.nc"
    if not fpath.exists():
        print(f"[static] Missing file for {year}: {fpath}")
        return None

    ds = xr.open_dataset(fpath)

    if phase == "FS":
        var_prefix = "advance"
    elif phase == "MS":
        var_prefix = "retreat"
    else:
        ds.close()
        raise ValueError("phase must be 'FS' or 'MS'")

    varname = f"{var_prefix}_{year}"
    if varname not in ds:
        print(f"[static] {varname} not in {fpath}; vars={list(ds.data_vars)}")
        ds.close()
        return None

    da = ds[varname].load()
    ds.close()

    if not np.any(np.isfinite(da.values)):
        print(f"[static] All-NaN {varname} in {fpath}, skipping year {year}")
        return None

    return da


def load_dynamic_year(phase: str, year: int) -> xr.DataArray | None:
    if phase == "FS":
        ddir = DYN_DIR_FS
    elif phase == "MS":
        ddir = DYN_DIR_MS
    else:
        raise ValueError("phase must be 'FS' or 'MS'")

    fpath = ddir / f"{phase}_{year}.nc"
    if not fpath.exists():
        print(f"[dynamic] Missing file for {phase} {year}: {fpath}")
        return None

    ds = xr.open_dataset(fpath)
    if phase not in ds:
        print(f"[dynamic] {phase} not in {fpath}; vars={list(ds.data_vars)}")
        ds.close()
        return None

    da = ds[phase].load()
    ds.close()

    if not np.any(np.isfinite(da.values)):
        print(f"[dynamic] All-NaN {phase} in {fpath}, skipping year {year}")
        return None

    return da


# ---------------------------------------------------------------------
# CORE COMPUTATION
# ---------------------------------------------------------------------

def compute_series_clim_anom(
    phase: str,
    loader: Callable[[str, int], xr.DataArray | None],
    year_start: int,
    year_end: int,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray | None, xr.DataArray | None]:
    """
    Compute climatology and anomalies for a phase.

    Returns:
      clim, anom, clim_dsa, anom_dsa

    Where:
      - clim/anom are in calendar DOY space for both FS and MS (backward compatible)
      - clim_dsa/anom_dsa are only computed for MS (days since Aug 15), else None
    """
    years = list(range(year_start, year_end + 1))

    arrays = []
    valid_years = []

    for y in years:
        da = loader(phase, y)
        if da is None:
            continue

        if "y" in da.dims and "x" in da.dims:
            da = da.transpose("y", "x")

        arrays.append(da.expand_dims(year=[y]))
        valid_years.append(y)

    if not arrays:
        raise ValueError(f"No valid years found for phase={phase} with given loader.")

    all_da = xr.concat(arrays, dim="year").assign_coords(year=("year", valid_years))

    # Calendar DOY climatology/anomaly (kept as-is)
    clim = all_da.mean("year", skipna=True)
    anom = all_da - clim

    # For MS only: compute continuous season-relative version
    clim_dsa = None
    anom_dsa = None
    if phase == "MS":
        all_da_dsa = ms_to_days_since_aug15(all_da)
        clim_dsa = all_da_dsa.mean("year", skipna=True)
        anom_dsa = all_da_dsa - clim_dsa

    return clim, anom, clim_dsa, anom_dsa


def write_clim_anom(
    phase: str,
    method: str,
    clim: xr.DataArray,
    anom: xr.DataArray,
    clim_dsa: xr.DataArray | None = None,
    anom_dsa: xr.DataArray | None = None,
) -> None:
    """
    Save climatology + anomalies to NetCDF in OUT_DIR.

    For MS, also writes *_dsa variables (days since Aug 15).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clim_name = f"{phase}_{method}_clim"
    anom_name = f"{phase}_{method}_anom"

    clim_ds = clim.to_dataset(name=clim_name)
    anom_ds = anom.to_dataset(name=anom_name)

    if phase == "MS" and clim_dsa is not None and anom_dsa is not None:
        clim_ds[f"{clim_name}_dsa"] = clim_dsa
        clim_ds[f"{clim_name}_dsa"].attrs["units"] = "days since Aug 15"
        clim_ds[f"{clim_name}_dsa"].attrs["description"] = (
            "Melt start in a continuous seasonal coordinate to avoid calendar year wrap artifacts"
        )

        anom_ds[f"{anom_name}_dsa"] = anom_dsa
        anom_ds[f"{anom_name}_dsa"].attrs["units"] = "days"
        anom_ds[f"{anom_name}_dsa"].attrs["description"] = (
            "Anomalies of melt start computed in days-since-Aug15 space"
        )

    clim_path = OUT_DIR / f"{phase}_{method}_climatology.nc"
    anom_path = OUT_DIR / f"{phase}_{method}_anomalies.nc"

    print(f"Saving climatology -> {clim_path}")
    clim_ds.to_netcdf(clim_path)

    print(f"Saving anomalies   -> {anom_path}")
    anom_ds.to_netcdf(anom_path)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir  : {OUT_DIR}")
    print(f"Years       : {YEAR_START}–{YEAR_END}")
    print(f"MS wrap     : days since Aug 15 (DOY {AUG15_DOY})")

    for method, loader in [("static", load_static_year), ("dynamic", load_dynamic_year)]:
        for phase in ["FS", "MS"]:
            print(f"\n=== {phase} ({method}) ===")
            clim, anom, clim_dsa, anom_dsa = compute_series_clim_anom(
                phase=phase,
                loader=loader,
                year_start=YEAR_START,
                year_end=YEAR_END,
            )
            write_clim_anom(phase, method, clim, anom, clim_dsa, anom_dsa)


if __name__ == "__main__":
    main()
