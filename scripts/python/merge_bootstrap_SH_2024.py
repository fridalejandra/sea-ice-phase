import os
import glob
import xarray as xr

# ---- CONFIG ---- #
SH_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/test_downloads/"
MERGED_FILE = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_extended_SH.nc"

# ---- LOAD VALID SH FILES ---- #
files = sorted(glob.glob(os.path.join(SH_DIR, "*S25km*.nc")))
datasets = []

for f in files:
    try:
        ds = xr.open_dataset(f)
        icecon_var = [v for v in ds.data_vars if v.endswith("_ICECON")][0]
        da = ds[icecon_var]
        datasets.append(da)
    except Exception as e:
        print(f"❌ Skipping {f}: {e}")

# ---- MERGE ---- #
if datasets:
    merged = xr.concat(datasets, dim="time")
    merged = merged.sortby("time")
    merged.to_dataset(name=icecon_var).to_netcdf(MERGED_FILE)
    print("✅ Merged SH dataset written to", MERGED_FILE)
else:
    print("⚠️ No valid SH datasets to merge.")
