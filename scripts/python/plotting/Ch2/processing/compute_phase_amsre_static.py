import xarray as xr
import numpy as np
import os
from tqdm import tqdm
import gc
import calendar

# === CONFIGURATION === #
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/SIC_07132012_04082025_merged.nc"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/AMSRE_phase/"
CONC_VAR = "SI_12km_SH_ICECON_DAY_SpPolarGrid12km"
THRESHOLD = 15  # AMSRE SIC is in percent
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
ice = ice.where(ice < 110)  # mask land and missing

all_years = np.unique(ds.time.dt.year.values)
x, y = ds.x.size, ds.y.size

# === MAIN LOOP === #
for year in tqdm(all_years[:-1], desc="Processing AMSRE phase years"):
    next_year = year + 1

    # Define cross-year windows
    retreat_start = f"{year}-08-25"
    retreat_end = f"{next_year}-02-29" if calendar.isleap(next_year) else f"{next_year}-02-28"
    advance_start = f"{year}-03-25"
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
                rt = ts_r.where((doy_retreat >= 237) | (doy_retreat <= 59), drop=True)  # Aug 25–Feb 28/29
                if (rt > THRESHOLD).any():
                    idx_rt = find_first_event(rt, THRESHOLD, WINDOW, above=False)
                    if not np.isnan(idx_rt):
                        retreat[j, i] = rt.time[idx_rt].dt.dayofyear.item()

                # --- ADVANCE ---
                ad = ts_a.where((doy_advance >= 84) & (doy_advance <= 258), drop=True)  # Mar 25–Sep 15
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
            "description": f"AMSRE Advance & Retreat | THRESHOLD={THRESHOLD}, WINDOW={WINDOW}, Year={year}"
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"seaice_phases_AMSRE_{year}.nc")
    out_ds.to_netcdf(out_file)
    print(f"✅ Saved {out_file}")

    del advance, retreat, out_ds, data_retreat, data_advance
    gc.collect()

