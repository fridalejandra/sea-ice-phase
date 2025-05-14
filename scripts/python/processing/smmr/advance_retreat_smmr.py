import xarray as xr
import numpy as np
import os
import calendar
from tqdm import tqdm
import gc

# === CONFIGURATION === #
INPUT_FILE = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/merged/SMMR_merged_1979_06302024.nc"
OUTPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/SMMR_phase/"
CONC_VAR = "N07_ICECON"
THRESHOLD = 0.15  # SMMR is 0-1
WINDOW = 5

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
ice = ds[CONC_VAR].astype("float32")
ice = ice.where(ice < 1.1)  # mask land and missing

all_years = np.unique(ds.time.dt.year.values)
x, y = ds.x.size, ds.y.size

# === MAIN LOOP === #
for year in tqdm(all_years[:-1], desc="Processing SMMR phase years"):
    next_year = year + 1

    # Define windows
    retreat_start = f"{year}-08-15"
    retreat_end = f"{next_year}-02-29" if calendar.isleap(next_year) else f"{next_year}-02-28"
    advance_start = f"{year}-02-01"
    advance_end = f"{year}-09-15"

    try:
        data_retreat = ice.sel(time=slice(retreat_start, retreat_end))
        data_advance = ice.sel(time=slice(advance_start, advance_end))
    except KeyError:
        continue

    if data_retreat.time.size < 60 or data_advance.time.size < 60:
        continue

    doy_retreat = data_retreat.time.dt.dayofyear
    doy_advance = data_advance.time.dt.dayofyear

    advance = np.full((y, x), np.nan)
    retreat = np.full((y, x), np.nan)

    for j in range(y):
        for i in range(x):
            ts_r = data_retreat[:, j, i]
            ts_a = data_advance[:, j, i]
            if ts_r.isnull().all() and ts_a.isnull().all():
                continue

            try:
                # --- RETREAT ---
                rt = ts_r.where((doy_retreat >= 227) | (doy_retreat <= 60), drop=True)
                if (rt > THRESHOLD).any():
                    idx_rt = find_first_event(rt, THRESHOLD, WINDOW, above=False)
                    if not np.isnan(idx_rt):
                        retreat[j, i] = rt.time[idx_rt].dt.dayofyear.item()

                # --- ADVANCE ---
                ad = ts_a.where((doy_advance >= 30) & (doy_advance <= 260), drop=True)
                if (ad > THRESHOLD).any():
                    idx_ad = find_first_event(ad, THRESHOLD, WINDOW, above=True)
                    if not np.isnan(idx_ad):
                        advance[j, i] = ad.time[idx_ad].dt.dayofyear.item()

            except Exception:
                continue

    out_ds = xr.Dataset(
        {
            f"advance_{year}": (("y", "x"), advance),
            f"retreat_{year}": (("y", "x"), retreat),
        },
        coords={"x": ds.x, "y": ds.y},
        attrs={
            "description": f"SMMR Advance & Retreat | THRESHOLD={THRESHOLD}, WINDOW={WINDOW}, Year={year}"
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"seaice_phases_SMMR_{year}.nc")
    out_ds.to_netcdf(out_file)
    print(f"✅ Saved {out_file}")

    del advance, retreat, out_ds, data_retreat, data_advance
    gc.collect()
