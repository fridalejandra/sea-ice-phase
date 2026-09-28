"""
compute_dsic_daily.py

Compute daily sea ice concentration TENDENCY (ΔSIC = SIC_t+1 - SIC_t) from
the regridded Bootstrap SIC on the EASE grid, and save as a NetCDF for
mapping with plot_monthly_maps.py.

This is the gridded version of the sector-level ΔSIA used in the regressions.
Mapping var(ΔSIC) pre vs post shows WHERE the system got quieter — the spatial
expression of the 30-70% variance collapse seen in the sector-mean table.
"""

import numpy as np
import xarray as xr

SIC_PATH = "sic_bootstrap_on_ease_sh.nc"
OUT_PATH = "dsic_daily_on_ease_sh.nc"
SIC_VAR = "sic"

def main():
    ds = xr.open_dataset(SIC_PATH)
    sic = ds[SIC_VAR]
    print(f"Loaded SIC: {dict(sic.sizes)}")

    # daily difference: SIC(t+1) - SIC(t)
    dsic = sic.diff(dim="time")
    dsic.name = "dsic"
    dsic.attrs["long_name"] = "daily sea ice concentration tendency"
    dsic.attrs["units"] = "fraction/day"
    dsic.attrs["note"] = "SIC(t+1) - SIC(t), forward difference"

    print(f"ΔSIC computed: {dict(dsic.sizes)}")
    print(f"  min={float(dsic.min()):.4f}, max={float(dsic.max()):.4f}")
    print(f"  median |ΔSIC| = {float(np.nanmedian(np.abs(dsic.values))):.4f}")

    dsic.to_netcdf(OUT_PATH)
    print(f"\n-> {OUT_PATH}")
    print("NEXT: add to plot_monthly_maps.py VARIABLES dict:")
    print('  "dsic": ("dsic_daily_on_ease_sh.nc", "dsic", "RdBu_r", "fraction/day", False),')

if __name__ == "__main__":
    main()