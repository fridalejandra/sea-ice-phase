# merge_smmr.py
import os, glob, numpy as np, xarray as xr
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

RAW_DIR    = "/user/geog/falejandraperez/sea-ice-phase/data/smmr/raw"
MERGED_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged"
TEMP_DIR   = f"{MERGED_DIR}/smmr_yearly"
today      = datetime.today().strftime("%m%d%Y")
OUT_FILE   = f"{MERGED_DIR}/SMMR_merged_19781101_{today}.nc"
LATEST     = f"{MERGED_DIR}/merged_bootstrap_SH_latest.nc"
MIN_SIZE   = 30_000

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

# only NSIDC granule files — exclude merged_YYYY.nc files
all_files  = sorted(glob.glob(os.path.join(RAW_DIR, "NSIDC0079_*.nc")))
data_files = [f for f in all_files if os.path.getsize(f) >= MIN_SIZE]

# extract year from filename: NSIDC0079_SEAICE_PS_S25km_YYYYMMDD_v4.0.nc
# date field is [4] = "YYYYMMDD" or "YYYYMM" (monthly — skip those)
def get_year(f):
    date_str = os.path.basename(f).split("_")[4]
    if len(date_str) == 8:   # daily YYYYMMDD
        return date_str[:4]
    return None   # monthly — skip

data_files = [f for f in data_files if get_year(f) is not None]
years = sorted(set(get_year(f) for f in data_files))
print(f"Found {len(data_files)} daily data files across {len(years)} years")

# step 1: merge each year
yearly_files = []
for year in tqdm(years, desc="Merging by year"):
    out_year = os.path.join(TEMP_DIR, f"SMMR_{year}.nc")
    if os.path.exists(out_year):
        yearly_files.append(out_year)
        continue
    yr_files = [f for f in data_files if get_year(f) == year]
    if not yr_files:
        continue
    try:
        ds = xr.open_mfdataset(yr_files, combine="nested",
                               concat_dim="time", parallel=False)
        ds = ds.sortby("time")
        ds.to_netcdf(out_year)
        ds.close()
        yearly_files.append(out_year)
    except Exception as e:
        print(f"  Failed {year}: {e}")

print(f"Created {len(yearly_files)} yearly files")

# step 2: concatenate all yearly files
print("Concatenating yearly files...")
ds_all = xr.open_mfdataset(yearly_files, combine="nested",
                            concat_dim="time", parallel=False)
ds_all = ds_all.sortby("time")
_, idx = np.unique(ds_all.time.values, return_index=True)
ds_all = ds_all.isel(time=idx)

print(f"Time range: {ds_all.time.values[0]} → {ds_all.time.values[-1]}")
print(f"Total timesteps: {ds_all.dims['time']}")

print(f"Saving to {OUT_FILE}...")
ds_all.to_netcdf(OUT_FILE)
ds_all.close()

if os.path.islink(LATEST) or os.path.exists(LATEST):
    os.remove(LATEST)
os.symlink(OUT_FILE, LATEST)
print(f"Symlink updated → {os.path.basename(OUT_FILE)}")
print("Done.")
