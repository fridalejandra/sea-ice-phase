"""
mask_and_check_bootstrap.py

The Bootstrap SIC is stored as a FRACTION (0-1), not percent, and uses flag
values ABOVE 1.0 (e.g. 1.2) for pole hole / land / coast / missing. Those
flags must be masked to NaN BEFORE regridding, or conservative regridding
blends them into real cells (that's why the regridded max came out ~10 and
the budget test produced all-NaN interior stats).

This script:
  1. loads bootstrap_sic_with_latlon.nc
  2. masks everything > 1.0 (and optionally exact known flag values) to NaN
  3. reports how many cells were flagged, so you can see it's sensible
  4. writes bootstrap_sic_masked_latlon.nc  (lat/lon preserved)

Then re-run the regrid pointing SIC_PATH at this masked file.
"""

import numpy as np
import xarray as xr

IN_PATH = "bootstrap_sic_with_latlon.nc"
OUT_PATH = "bootstrap_sic_masked_latlon.nc"
SIC_VAR = "N07_ICECON"

# Bootstrap SH flag conventions vary by product version; masking anything
# strictly > 1.0 catches the 1.2 sentinel seen in this file. If you know the
# exact flag values, list them here too for clarity/logging.
KNOWN_FLAGS = [1.2]

ds = xr.open_dataset(IN_PATH)
A = ds[SIC_VAR]

print(f"Before masking: min={float(A.min()):.3f}, max={float(A.max()):.3f}")
for f in KNOWN_FLAGS:
    n = int((np.isclose(A.values, f)).sum())
    print(f"  cells == {f}: {n:,}")
n_gt1 = int((A.values > 1.0).sum())
print(f"  cells > 1.0 total: {n_gt1:,} of {A.size:,} "
      f"({100*n_gt1/A.size:.2f}%)")

# mask: valid concentration is 0..1 inclusive; everything else -> NaN
A_masked = A.where((A >= 0.0) & (A <= 1.0))

print(f"After masking:  min={float(A_masked.min()):.3f}, "
      f"max={float(A_masked.max()):.3f}, "
      f"n valid={int(A_masked.notnull().sum()):,}")

# sanity: on a mid-winter day there should be a LOT of high-concentration pack
mid = A_masked.sel(time="2010-07-15") if "2010-07-15" in \
    np.datetime_as_string(A_masked["time"].values, unit="D") else \
    A_masked.isel(time=len(A_masked.time)//2)
n_pack = int((mid > 0.95).sum())
print(f"\nMid-winter check: {n_pack:,} cells with A>0.95 "
      f"(should be tens of thousands for SH winter pack).")

ds[SIC_VAR] = A_masked
ds.to_netcdf(OUT_PATH)
print(f"\n-> {OUT_PATH}")
print("NEXT: in regrid_sic_to_ease.py set SIC_PATH = this file, then rerun "
      "the regrid. Delete the cached weights first if the grid didn't change: "
      "the weights are fine, but the DATA masking means you want a clean apply. "
      "Actually the weights are unaffected by data values -- just rerun; "
      "xESMF will reuse weights and apply to the masked data.")
