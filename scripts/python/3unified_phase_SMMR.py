import xarray as xr
import numpy as np
from tqdm import tqdm
import os

# === CONFIGURATION === #

# ---- INPUT / OUTPUT PATHS ---- #
# LOCAL:
# INPUT_FILE = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_SH_1979_06302024.nc"
# OUTPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/test_downloads/"

# CLUSTER:
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_SH_1979_06302024.nc"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/data/bootstrap_smmr/test_downloads/"

CONC_VAR = "N07_ICECON"
THRESHOLD = 0.15
WINDOW = 5  # days

ADVANCE_START_DOY = 30    # Jan 30
ADVANCE_END_DOY = 180     # Jun 29
RETREAT_START_DOY = 240   # Aug 27
RETREAT_END_DOY = 365     # Dec 31

# === HELPER FUNCTION === #
def find_first_event(data, threshold, window, above=True):
    condition = data > threshold if above else data < threshold
    rolling = condition.rolling(time=window).construct("window")
    hits = rolling.all("window")
    if not hits.any():
        return np.nan
    return int(hits.argmax("time").item())

# === LOAD DATA === #
ds = xr.open_dataset(INPUT_FILE)
ice = ds[CONC_VAR].astype("float32") * 1.0
ice = ice.where(ice < 1.1)  # remove land/missing

years = np.unique(ds.time.dt.year.values)
x, y = ds.x.size, ds.y.size

for year in tqdm(years, desc="Processing years"):
    year_data = ice.sel(time=str(year))
    doy = year_data.time.dt.dayofyear

    advance = np.full((y, x), np.nan)
    retreat = np.full((y, x), np.nan)

    for j in range(y):
        for i in range(x):
            ts = year_data[:, j, i]
            if ts.isnull().all():
                continue

            try:
                def sub_doy(start, end): return ts[(doy >= start) & (doy <= end)]

                # Retreat
                rt = sub_doy(RETREAT_START_DOY, RETREAT_END_DOY)
                idx_rt = find_first_event(rt, THRESHOLD, WINDOW, above=False)
                retreat[j, i] = rt.time[idx_rt].dt.dayofyear.item() if not np.isnan(idx_rt) else np.nan

                # Advance
                ad = sub_doy(ADVANCE_START_DOY, ADVANCE_END_DOY)
                idx_ad = find_first_event(ad, THRESHOLD, WINDOW, above=True)
                advance[j, i] = ad.time[idx_ad].dt.dayofyear.item() if not np.isnan(idx_ad) else np.nan

            except Exception:
                continue

    out_ds = xr.Dataset(
        {
            "advance": (("y", "x"), advance),
            "retreat": (("y", "x"), retreat),
        },
        coords={"x": ds.x, "y": ds.y},
        attrs={"description": f"SMMR advance/retreat timing (threshold: {THRESHOLD}, window: {WINDOW}) for {year}"}
    )

    out_file = os.path.join(OUTPUT_DIR, f"seaice_phases_SMMR_{year}.nc")
    out_ds.to_netcdf(out_file)
    print(f"✅ Saved {out_file}")
