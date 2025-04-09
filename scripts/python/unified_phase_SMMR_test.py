import xarray as xr
import numpy as np
from tqdm import tqdm
import os

# === SETTINGS === #
INPUT_FILE = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/merged_bootstrap_SH_1979_06302024.nc"
OUTPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/bootstrap_smmr/test_downloads/"
CONC_VAR = "N07_ICECON"
THRESHOLD = 0.15
WINDOW = 5  # consecutive days

# Time search windows (DOY)
RETREAT_START_DOY = 240  # August 27
RETREAT_END_DOY = 365
ADVANCE_START_DOY = 30   # Jan 30
ADVANCE_END_DOY = 180

# For early melt and late freeze (optional refinements)
EARLY_MELT_START_DOY = 120
EARLY_MELT_END_DOY = 180
LATE_FREEZE_START_DOY = 0
LATE_FREEZE_END_DOY = 90

# === HELPER FUNCTION === #
def find_first_event(data, threshold, window, above=True):
    """Find the first day where condition is met for `window` consecutive days."""
    condition = data > threshold if above else data < threshold
    rolling = condition.rolling(time=window).construct("window")
    hits = rolling.all("window")
    result = hits.argmax("time").item()
    if not hits.any():
        return np.nan
    return int(result)

# === LOAD DATA === #
ds = xr.open_dataset(INPUT_FILE)
ice = ds[CONC_VAR].astype("float32") * 1.0  # unpack
ice = ice.where(ice < 1.1)  # remove land/missing

# === PREPARE OUTPUT === #
years = np.unique(ds.time.dt.year.values)
x, y = ds.x.size, ds.y.size

for year in tqdm(years, desc="Processing years"):
    # Select full year slice
    year_data = ice.sel(time=str(year))

    # Calculate day of year
    doy = year_data.time.dt.dayofyear

    # Prepare arrays to store output
    early_melt = np.full((y, x), np.nan)
    retreat = np.full((y, x), np.nan)
    advance = np.full((y, x), np.nan)
    late_freeze = np.full((y, x), np.nan)

    for j in range(y):
        for i in range(x):
            ts = year_data[:, j, i]
            if ts.isnull().all():
                continue

            try:
                # Subset for each search window
                def sub_doy(start, end): return ts[(doy >= start) & (doy <= end)]

                # Early Melt (first below threshold in spring)
                em = sub_doy(EARLY_MELT_START_DOY, EARLY_MELT_END_DOY)
                early_melt[j, i] = em.time[find_first_event(em, THRESHOLD, WINDOW, above=False)].dt.dayofyear.item()

                # Retreat (main melt period)
                rt = sub_doy(RETREAT_START_DOY, RETREAT_END_DOY)
                retreat[j, i] = rt.time[find_first_event(rt, THRESHOLD, WINDOW, above=False)].dt.dayofyear.item()

                # Advance (main freeze-up)
                ad = sub_doy(ADVANCE_START_DOY, ADVANCE_END_DOY)
                advance[j, i] = ad.time[find_first_event(ad, THRESHOLD, WINDOW, above=True)].dt.dayofyear.item()

                # Late Freeze (second freeze-up attempt)
                lf = sub_doy(LATE_FREEZE_START_DOY, LATE_FREEZE_END_DOY)
                late_freeze[j, i] = lf.time[find_first_event(lf, THRESHOLD, WINDOW, above=True)].dt.dayofyear.item()

            except:
                continue

    # === SAVE OUTPUT === #
    out_ds = xr.Dataset(
        {
            "early_melt": (("y", "x"), early_melt),
            "retreat": (("y", "x"), retreat),
            "advance": (("y", "x"), advance),
            "late_freeze": (("y", "x"), late_freeze),
        },
        coords={"x": ds.x, "y": ds.y},
        attrs={"description": f"Sea ice phase dates (threshold: {THRESHOLD}, window: {WINDOW} days) for year {year}"}
    )
    out_ds.to_netcdf(os.path.join(OUTPUT_DIR, f"seaice_phases_SMMR_{year}.nc"))
    print(f"✅ Saved seaice_phases_SMMR_{year}.nc")
