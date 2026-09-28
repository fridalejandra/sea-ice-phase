import xarray as xr
import os
import numpy as np
import glob
from datetime import datetime

BASE        = "/user/geog/falejandraperez/sea-ice-phase/data"
DAILY_DIR   = f"{BASE}/amsre/daily_nc"
EXISTING    = f"{BASE}/merged/AMSRE_merged_07132012_04082025.nc"
OUT_FILE    = f"{BASE}/merged/AMSRE_merged_07132012_latest.nc"

# load new daily files
new_files = sorted(glob.glob(os.path.join(DAILY_DIR, "SIC_*.nc")))
print(f"Found {len(new_files)} new daily files")
ds_new = xr.open_mfdataset(new_files, combine="nested", concat_dim="time")
ds_new = ds_new.sortby("time")
print(f"New data: {ds_new.time.values[0]} → {ds_new.time.values[-1]}")

# load existing merged file
print(f"Loading existing: {EXISTING}")
ds_old = xr.open_dataset(EXISTING)
print(f"Existing data: {ds_old.time.values[0]} → {ds_old.time.values[-1]}")

# concatenate and remove duplicates
ds_all = xr.concat([ds_old, ds_new], dim="time")
ds_all = ds_all.sortby("time")
_, idx = np.unique(ds_all.time.values, return_index=True)
ds_all = ds_all.isel(time=idx)
print(f"Combined: {ds_all.time.values[0]} → {ds_all.time.values[-1]}, {len(ds_all.time)} days")

ds_all.to_netcdf(OUT_FILE)
print(f"Saved to {OUT_FILE}")
