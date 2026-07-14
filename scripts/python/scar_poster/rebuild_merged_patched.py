import xarray as xr, numpy as np, glob, os, re

YEARLY_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/smmr_yearly"
OUT    = "/user/geog/falejandraperez/sea-ice-phase/data/merged/SMMR_merged_19781101_20251231_complete.nc"
LATEST = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
FILL_VALUE = -9999.0

# dynamically detect any *_ICECON variable rather than a hardcoded sensor
# list, so a new/unexpected sensor prefix can't silently cause a dropped year
ICECON_PATTERN = re.compile(r".*_ICECON$", re.IGNORECASE)

yearly_files = sorted(glob.glob(os.path.join(YEARLY_DIR, "SMMR_*.nc")))
print(f"Found {len(yearly_files)} yearly files")

yearly_combined = []
for f in yearly_files:
    year = os.path.basename(f).split('_')[1].split('.')[0]
    try:
        ds = xr.open_dataset(f)
        n = ds.sizes.get('time', 0)
        if n == 0:
            ds.close()
            continue

        present = [v for v in ds.variables if ICECON_PATTERN.match(v)]
        if not present:
            print(f"  WARNING {year}: no *_ICECON variable found - SKIPPING YEAR")
            ds.close()
            continue

        # combine all sensor variables into one SIC field, preferring the
        # first non-NaN value across sensors for each pixel/day
        sic = ds[present[0]].copy()
        for v in present[1:]:
            sic = sic.where(sic.notnull(), ds[v])

        ds_out = xr.Dataset(
            {"N07_ICECON": sic},
            coords={"time": ds.time, "y": ds.y, "x": ds.x}
        )
        yearly_combined.append(ds_out)
        print(f"  {year}: {n} timesteps, sensors={present}")
        ds.close()
    except Exception as e:
        print(f"  FAILED {year}: {e}")

print(f"\nConcatenating {len(yearly_combined)} years...")
ds_all = xr.concat(yearly_combined, dim="time").sortby("time")
_, idx = np.unique(ds_all.time.values, return_index=True)
ds_all = ds_all.isel(time=idx)

years = np.unique(ds_all.time.dt.year.values)
print(f"Years: {years}")
print(f"Total timesteps: {ds_all.sizes['time']}")
print(f"Time range: {ds_all.time.values[0]} → {ds_all.time.values[-1]}")

# --- sanity check: make sure NaNs are actually preserved as NaN, not 0 ---
sic_arr = ds_all["N07_ICECON"].values
n_nan = np.isnan(sic_arr).sum()
n_zero = (sic_arr == 0).sum()
print(f"\nValidity check: {n_nan} NaN cells, {n_zero} exact-zero cells "
      f"(out of {sic_arr.size} total)")
if n_nan == 0:
    print("  WARNING: zero NaNs found - if you expect missing data to exist, "
          "this may indicate NaNs are still being lost on write. Investigate "
          "before trusting this file.")

print(f"\nSaving to {OUT}...")
encoding = {
    var: {"dtype": "float32", "_FillValue": FILL_VALUE}
    for var in ds_all.data_vars
    if np.issubdtype(ds_all[var].dtype, np.floating)
}
ds_all.to_netcdf(OUT, encoding=encoding)

if os.path.islink(LATEST) or os.path.exists(LATEST):
    os.remove(LATEST)
os.symlink(OUT, LATEST)
print(f"Done. Symlink → {os.path.basename(OUT)}")