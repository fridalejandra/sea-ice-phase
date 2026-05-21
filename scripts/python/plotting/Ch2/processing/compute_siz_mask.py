#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_seasonal_ice_zone_mask.py

Builds an Antarctic *seasonal ice zone* mask from the merged Bootstrap SIC file:

  /user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_SH_1979_20251001.nc

Definition (Option C you chose):
  - "Summer"  = February–March
  - "Winter"  = September–October
  - Seasonal ice zone if:
        mean summer SIC < SUMMER_MAX
    AND mean winter SIC > WINTER_MIN
    AND mean winter SIC < PERENNIAL_MAX
    AND SIC is finite

Thresholds are in *physical* units after scale_factor/add_offset
(i.e. assuming xarray decode_cf gives SIC in 0–1).

Outputs:
  results/masks/seasonal_ice_zone_mask.nc
with a boolean DataArray 'seasonal_ice_zone' on (y,x).

You can then multiply this with your valid_ocean mask and/or pass it
into the anomaly plotting scripts.
"""

from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")

SIC_PATH = PROJECT_ROOT / "data" / "bootstrap_smmr" / "merged_bootstrap_SH_1979_20251001.nc"

# Thresholds in SIC fraction (0–1); change if your SIC is 0–100 etc.
WINTER_MIN     = 0.15   # must be at least 15% in winter
SUMMER_MAX     = 0.15   # must be < 15% in summer
PERENNIAL_MAX  = 0.90   # exclude nearly-perennial pack

# Months for seasonal definition (Option C)
SUMMER_MONTHS = [2, 3]   # Feb–Mar
WINTER_MONTHS = [9, 10]  # Sep–Oct

OUT_MASK_PATH = PROJECT_ROOT / "results" / "masks" / "seasonal_ice_zone_mask.nc"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def detect_sic_var(ds: xr.Dataset) -> str:
    """
    Try to guess the SIC variable name.

    Preference order:
      1) variable with standard_name 'sea_ice_area_fraction'
      2) first variable whose name contains 'ICECON' (Bootstrap-like)
      3) first data_var as last resort
    """
    # 1. standard_name
    for v in ds.data_vars:
        std = ds[v].attrs.get("standard_name", "").lower()
        if std == "sea_ice_area_fraction":
            print(f"[mask] Using SIC variable by standard_name: {v}")
            return v

    # 2. name contains ICECON
    icecon_like = [v for v in ds.data_vars if "icecon" in v.lower()]
    if icecon_like:
        print(f"[mask] Using SIC variable by name pattern: {icecon_like[0]}")
        return icecon_like[0]

    # 3. fallback: first variable
    v0 = list(ds.data_vars)[0]
    print(f"[mask] WARNING: falling back to first variable: {v0}")
    return v0


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"[mask] Opening SIC file: {SIC_PATH}")
    ds = xr.open_dataset(SIC_PATH, chunks={"time": 365})

    sic_var = detect_sic_var(ds)
    sic = ds[sic_var]

    # Ensure time is decoded to datetime
    if not np.issubdtype(sic["time"].dtype, np.datetime64):
        sic["time"] = xr.decode_cf(ds).time  # crude; if decode_cf failed earlier

    # Remove obviously bad values (Bootstrap sometimes uses >1 for flags)
    sic = sic.where(np.isfinite(sic))
    sic = sic.where((sic >= 0.0) & (sic <= 1.5))  # loose upper bound

    # Monthly climatology over all years
    print("[mask] Computing monthly climatology...")
    clim_month = sic.groupby("time.month").mean("time", skipna=True)

    # Summer & winter mean SIC
    sic_summer = clim_month.sel(month=SUMMER_MONTHS).mean("month")
    sic_winter = clim_month.sel(month=WINTER_MONTHS).mean("month")

    # Build masks
    summer_open   = sic_summer < SUMMER_MAX
    winter_iced   = sic_winter > WINTER_MIN
    not_perennial = sic_winter < PERENNIAL_MAX

    seasonal_mask = summer_open & winter_iced & not_perennial
    seasonal_mask = seasonal_mask & np.isfinite(sic_summer) & np.isfinite(sic_winter)

    seasonal_mask = seasonal_mask.rename("seasonal_ice_zone")
    seasonal_mask.attrs["description"] = (
        "Boolean seasonal ice zone mask based on Bootstrap SIC. "
        f"Summer months={SUMMER_MONTHS}, winter months={WINTER_MONTHS}, "
        f"thresholds: WINTER_MIN={WINTER_MIN}, SUMMER_MAX={SUMMER_MAX}, "
        f"PERENNIAL_MAX={PERENNIAL_MAX} (units of SIC fraction)."
    )

    # Save
    OUT_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mask] Saving mask to: {OUT_MASK_PATH}")
    seasonal_mask.to_netcdf(OUT_MASK_PATH)

    print("[mask] Done.")


if __name__ == "__main__":
    main()
