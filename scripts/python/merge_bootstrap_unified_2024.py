import earthaccess
import xarray as xr
import os
import datetime
import pandas as pd

# ======== CONFIGURATION ========
EXISTING_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/merged_bootsstrap.nc"
DOWNLOAD_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/test_downloads/"
MERGED_OUTPUT = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_extended_SH.nc"
ORIGINAL_VAR = "F17_ICECON"
RENAME_TO = "N07_ICECON"

# ======== AUTHENTICATE ========
earthaccess.login(strategy="netrc")

# ======== DETERMINE DATE RANGE ========
existing_ds = xr.open_dataset(EXISTING_FILE)
last_date = str(existing_ds.time.max().values)[:10]
start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
print(f"Searching for granules from {start_date} to {end_date}...")

# ======== SEARCH ========
results = earthaccess.search_data(
    short_name="NSIDC-0079",
    version="4",
    temporal=(start_date, end_date),
    bounding_box=(-180, -90, 180, 90)
)

if not results:
    print("No new files found.")
    exit()

print(f"Found {len(results)} granules. Downloading...")
downloaded = earthaccess.download(results, local_path=DOWNLOAD_DIR)

# ======== FUNCTION TO MASK NH PIXELS ========
def mask_nh_from_y(ds, varname="F17_ICECON"):
    if "y" in ds[varname].dims:
        ds[varname].loc[dict(y=slice(448, None))] = np.nan
    return ds

# ======== LOAD, RENAME, MASK, PREPARE ========
new_datasets = []
for f in downloaded:
    try:
        ds = xr.open_dataset(f)
        if ORIGINAL_VAR in ds:
            ds = mask_nh_from_y(ds, varname=ORIGINAL_VAR)
            temp = ds[[ORIGINAL_VAR]].rename({ORIGINAL_VAR: RENAME_TO})
            new_datasets.append(temp)
    except Exception as e:
        print(f"⚠️ Error opening {f}: {e}")

if not new_datasets:
    print(f"No valid datasets with {ORIGINAL_VAR} found.")
    exit()

print("Merging with existing file...")

# ======== MERGE DATASETS ========
combined_new = xr.concat(new_datasets, dim="time")
combined_existing = existing_ds[[RENAME_TO]]
all_data = xr.concat([combined_existing, combined_new], dim="time")
all_data = all_data.sortby("time")
all_data = all_data.sel(time=~all_data.get_index("time").duplicated())

# ======== SLICE SH ONLY: y-dimension slicing ========
print("Applying final SH filter (y=0:448)...")
all_data = all_data.isel(y=slice(0, 448))

# Optional: clean encoding
all_data[RENAME_TO].encoding["_FillValue"] = -9999

# ======== SAVE TO FILE ========
all_data.to_netcdf(MERGED_OUTPUT)
print(f"✅ Southern Hemisphere only dataset saved to {MERGED_OUTPUT}")