import xarray as xr

"""
merge_smmr.py  –  Step 2
Merges two NetCDF files:
  1. ds_base : Stammerjohn 2008 historical record  (BASE_NC)
  2. ds_new  : newly downloaded / merged 2024-present granules  (MERGED_2024)

Produces a single time-sorted file saved to FINAL_MERGED, and updates
the stable symlink LATEST_MERGED so downstream scripts always find it.
"""

import sys
import os
import xarray as xr
from config import BASE_NC, MERGED_2024, FINAL_MERGED, LATEST_MERGED

# ---- CHECKS ---- #
for path, label in [(BASE_NC, "BASE_NC"), (MERGED_2024, "MERGED_2024")]:
    if not path.exists():
        print(f"{label} not found: {path}")
        sys.exit(1)

# ---- LOAD ---- #
print(f"Loading base file:  {BASE_NC}")
ds_base = xr.open_dataset(BASE_NC)

print(f"Loading new file:   {MERGED_2024}")
ds_new  = xr.open_dataset(MERGED_2024)

# ---- MATCH VARIABLE NAME ---- #
def get_icecon_var(ds, label):
    candidates = [v for v in ds.data_vars if v.endswith("_ICECON")]
    if not candidates:
        raise ValueError(f"No *_ICECON variable found in {label}. "
                         f"Available vars: {list(ds.data_vars)}")
    return candidates[0]

var_base = get_icecon_var(ds_base, "base file")
var_new  = get_icecon_var(ds_new,  "new file")

if var_base != var_new:
    print(f"Variable name mismatch: '{var_base}' vs '{var_new}'. Renaming new → base.")
    ds_new = ds_new.rename({var_new: var_base})

print(f"   Using variable: {var_base}")

# ---- CHECK FOR TIME OVERLAP ---- #
t_base = ds_base.time.values
t_new  = ds_new.time.values
overlap = set(t_base) & set(t_new)
if overlap:
    print(f"{len(overlap)} overlapping time steps found. "
          "Keeping base file values for those dates.")
    ds_new = ds_new.sel(time=~ds_new.time.isin(list(overlap)))

# ---- MERGE & SORT ---- #
print("Concatenating and sorting by time...")
merged = xr.concat([ds_base[var_base], ds_new[var_base]], dim="time")
merged = merged.sortby("time")

# ---- SAVE DATED FILE ---- #
FINAL_MERGED.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving merged file to: {FINAL_MERGED}")
merged.to_dataset(name=var_base).to_netcdf(str(FINAL_MERGED))

# ---- UPDATE SYMLINK ---- #
if LATEST_MERGED.is_symlink() or LATEST_MERGED.exists():
    LATEST_MERGED.unlink()
LATEST_MERGED.symlink_to(FINAL_MERGED)
print(f"🔗 Symlink updated: {LATEST_MERGED} → {FINAL_MERGED}")

print(f"\nMerge complete. Time range: "
      f"{str(merged.time.values[0])[:10]} → {str(merged.time.values[-1])[:10]}")