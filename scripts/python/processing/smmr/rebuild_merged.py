import xarray as xr, numpy as np, glob, os

YEARLY_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/merged/smmr_yearly"
OUT    = "/user/geog/falejandraperez/sea-ice-phase/data/merged/SMMR_merged_19781101_20251231_complete.nc"
LATEST = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"

SENSORS = ["N07_ICECON", "F08_ICECON", "F11_ICECON", "F13_ICECON", "F17_ICECON"]

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
        # combine all sensor variables into one SIC field
        present = [v for v in SENSORS if v in ds]
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

print(f"Saving to {OUT}...")
ds_all.to_netcdf(OUT)

if os.path.islink(LATEST) or os.path.exists(LATEST):
    os.remove(LATEST)
os.symlink(OUT, LATEST)
print(f"Done. Symlink → {os.path.basename(OUT)}")
