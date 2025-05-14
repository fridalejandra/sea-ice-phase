import xarray as xr
import os

DAILY_DIR = "data/amsre/daily_nc"
MERGED_FILE = "data/amsre/SIC_07132012_01012024.nc"

ds = xr.open_mfdataset(
    os.path.join(DAILY_DIR, "SIC_*.nc"),
    combine="nested",
    concat_dim="time",
    preprocess=lambda d: d.expand_dims("time"),
    parallel=True
)

ds = ds.sortby("time")  # Just in case
ds.to_netcdf(MERGED_FILE)
print(f"✅ Merged file saved to {MERGED_FILE}")
