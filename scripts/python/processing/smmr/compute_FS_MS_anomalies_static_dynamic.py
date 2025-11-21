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
  MS_static_climatology.nc
  FS_dynamic_climatology.nc
  MS_dynamic_climatology.nc

  FS_static_anomalies.nc       (FS_static_anom[year,y,x])
  MS_static_anomalies.nc
  FS_dynamic_anomalies.nc
  MS_dynamic_anomalies.nc
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Cluster project root – change if you ever run this elsewhere
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


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------

def load_static_year(phase: str, year: int) -> xr.DataArray | None:
    """
    Load static FS/MS for one year.

    phase: 'FS' or 'MS'
    year : e.g. 1980

    Returns DataArray [y,x] or None if not available.
    """
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
    """
    Load dynamic FS/MS for one year.

    phase: 'FS' or 'MS'
    """
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
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    For a given phase ('FS' or 'MS') and loader (static or dynamic),
    compute:

      - climatology[y,x]
      - anomalies[year,y,x]

    using all years in [year_start, year_end] where data exist.
    """
    years = list(range(year_start, year_end + 1))

    arrays = []
    valid_years = []

    for y in years:
        da = loader(phase, y)
        if da is None:
            continue

        # Ensure consistent dims order
        if "y" in da.dims and "x" in da.dims:
            da = da.transpose("y", "x")
        arrays.append(da.expand_dims(year=[y]))
        valid_years.append(y)

    if not arrays:
        raise ValueError(f"No valid years found for phase={phase} with given loader.")

    all_da = xr.concat(arrays, dim="year")
    all_da = all_da.assign_coords(year=("year", valid_years))

    clim = all_da.mean("year", skipna=True)
    anom = all_da - clim  # broadcast over 'year'

    return clim, anom


def write_clim_anom(
    phase: str,
    method: str,
    clim: xr.DataArray,
    anom: xr.DataArray,
) -> None:
    """
    Save climatology + anomalies to NetCDF in OUT_DIR.

    method: 'static' or 'dynamic'
    phase : 'FS' or 'MS'
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clim_name = f"{phase}_{method}_clim"
    anom_name = f"{phase}_{method}_anom"

    clim_ds = clim.to_dataset(name=clim_name)
    anom_ds = anom.to_dataset(name=anom_name)

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

    for method, loader in [("static", load_static_year), ("dynamic", load_dynamic_year)]:
        for phase in ["FS", "MS"]:
            print(f"\n=== {phase} ({method}) ===")
            clim, anom = compute_series_clim_anom(
                phase=phase,
                loader=loader,
                year_start=YEAR_START,
                year_end=YEAR_END,
            )
            write_clim_anom(phase, method, clim, anom)


if __name__ == "__main__":
    main()
