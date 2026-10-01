"""
compute_mean_ice_edge_climatology.py — all-time mean sea-ice concentration
on the native NSIDC polar-stereographic grid, cached small (~1MB) so
fig_02_sector_map.py (and any future map figure) never needs to touch the
13GB source file again.

Reads MERGED_ICECON_NC (data/merged/merged_bootstrap_SH_latest.nc), masks out
non-physical values (anything >1.0 -- after CF scale_factor decoding, missing
is 1.1 and land is 1.2, not real concentration), computes the all-time mean
per grid cell with dask (chunked by time so the full 13GB array is never
loaded into memory at once), and writes MEAN_ICE_CONC_NC.

This is a one-off cache-building step, like the compute/ scripts elsewhere in
this folder -- re-run it only if the source concentration file is updated.

Run from figures/compute/: python3 compute_mean_ice_edge_climatology.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # figures/compute -> figures

from ch3_config import MERGED_ICECON_NC, MEAN_ICE_CONC_NC

import numpy as np
import xarray as xr

if not os.path.exists(MERGED_ICECON_NC):
    raise SystemExit(f"Not found: {MERGED_ICECON_NC}")

print(f"Opening {MERGED_ICECON_NC} (dask-chunked by time, out-of-core)...")
ds = xr.open_dataset(MERGED_ICECON_NC, decode_times=False, chunks={"time": 365})
conc = ds["N07_ICECON"]
print(f"  shape: {conc.shape}  (time, y, x)")

# Values >1.0 are flags after CF scale_factor decoding (missing=1.1, land=1.2)
# -- not physical concentration -- mask them out before averaging.
valid = conc.where(conc <= 1.0)

print("Computing all-time mean (reads the whole file once, chunked by year)...")
mean_conc = valid.mean(dim="time", skipna=True).compute()

n_finite = int(np.isfinite(mean_conc.values).sum())
n_total = mean_conc.size
print(f"Grid: {mean_conc.shape}, finite cells: {n_finite}/{n_total}")
print(f"Mean concentration range: {np.nanmin(mean_conc.values):.4f} - {np.nanmax(mean_conc.values):.4f}")

out = xr.Dataset(
    {"mean_concentration": (("y", "x"), mean_conc.values)},
    coords={"y": ds["y"].values, "x": ds["x"].values},
    attrs={
        "description": (
            "All-time (1978-11-01 to 2025-12-31) mean sea ice concentration, "
            "Bootstrap SH, NSIDC polar stereographic 25km grid (EPSG:3412). "
            "Values >1.0 in the source (missing=1.1, land=1.2 after CF scale-"
            "factor decoding) were masked out of the mean. Built to draw a "
            "climatological mean ice-edge contour (0.15 threshold, see "
            "ch3_config.ICE_EDGE_CONCENTRATION) on fig_02_sector_map.py."
        ),
        "source_file": os.path.basename(MERGED_ICECON_NC),
        "crs_epsg": "3412",
        "built_by": "compute_mean_ice_edge_climatology.py",
    },
)

os.makedirs(os.path.dirname(MEAN_ICE_CONC_NC), exist_ok=True)
out.to_netcdf(MEAN_ICE_CONC_NC)
print(f"Wrote {MEAN_ICE_CONC_NC} ({os.path.getsize(MEAN_ICE_CONC_NC) / 1024:.1f} KB)")

ds.close()
print("Done.")
