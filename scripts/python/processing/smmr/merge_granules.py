"""
merge_granules.py  –  Step 1
Concatenates the new daily granule .nc files in GRANULE_DIR
into a single intermediate file (MERGED_NEW), sorted by time.
Only processes files from BASE_NC_END_DATE onward.
"""

import sys
import xarray as xr
from config import GRANULE_DIR, MERGED_NEW, BASE_NC_END_DATE, SIC_VAR

# ---- FIND FILES ---- #
all_files = sorted(GRANULE_DIR.glob("*.nc"))

# Filter to only files after the base record ends (by filename date)
def file_date(f):
    # filename format: NSIDC0079_SEAICE_PS_S25km_YYYYMMDD_v4.0.nc
    try:
        return f.stem.split("_")[4]  # YYYYMMDD
    except IndexError:
        return ""

cutoff = BASE_NC_END_DATE.replace("-", "")
new_files = [f for f in all_files if file_date(f) >= cutoff]

if not new_files:
    print(f"No granule files found in {GRANULE_DIR} at or after {BASE_NC_END_DATE}")
    sys.exit(1)

print(f"📂 Found {len(new_files)} new granule files (>= {BASE_NC_END_DATE})")
print(f"   First : {new_files[0].name}")
print(f"   Last  : {new_files[-1].name}")

# ---- OPEN & CONCATENATE ---- #
print("\nConcatenating granules...")
ds = xr.open_mfdataset(
    [str(f) for f in new_files],
    combine="by_coords",
    engine="h5netcdf",
    parallel=True,
)

# ---- IDENTIFY SIC VARIABLE ---- #
if SIC_VAR in ds.data_vars:
    sic_var = SIC_VAR
else:
    candidates = [v for v in ds.data_vars if v.endswith("_ICECON")]
    if not candidates:
        raise ValueError(f"No *_ICECON variable found. Available: {list(ds.data_vars)}")
    sic_var = candidates[0]
    print(f"SIC_VAR '{SIC_VAR}' not found; using '{sic_var}' instead.")

print(f"   Variable   : {sic_var}")

ds = ds.sortby("time")
print(f"   Time range : {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")
print(f"   Time steps : {ds.sizes['time']}")

# ---- SAVE ---- #
MERGED_NEW.parent.mkdir(parents=True, exist_ok=True)
print(f"\nSaving to: {MERGED_NEW}")
ds[[sic_var]].to_netcdf(str(MERGED_NEW))

print(f"\nGranule merge complete.")