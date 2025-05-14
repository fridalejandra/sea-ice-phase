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
THRESHOLD = 0.15
WINDOW = 5
YEAR = 2015  # ← Only this year will be processed

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

x, y = ds.x.size, ds.y.size
next_year = YEAR + 1

# === Define windows === #
retreat_start = f"{YEAR}-08-20"
retreat_end = f"{next_year}-02-29" if calendar.isleap(next_year) else f"{next_year}-02-28"
advance_start = f"{YEAR}-02-01"
advance_end = f"{YEAR}-10-01"

try:
    data_retreat = ice.sel(time=slice(retreat_start, retreat_end))
    data_advance = ice.sel(time=slice(advance_start, advance_end))
except KeyError:
    raise RuntimeError("Time slice failed. Check date ranges.")

if data_retreat.time.size < 60 or data_advance.time.size < 60:
    raise RuntimeError("Insufficient time steps for retreat or advance window.")

doy_retreat = data_retreat.time.dt.dayofyear
doy_advance = data_advance.time.dt.dayofyear

advance = np.full((y, x), np.nan)
retreat = np.full((y, x), np.nan)

# === MAIN LOOP === #
for j in tqdm(range(y), desc=f"Processing SMMR phase for {YEAR}"):
    for i in range(x):
        ts_r = data_retreat[:, j, i]
        ts_a = data_advance[:, j, i]
        if ts_r.isnull().all() and ts_a.isnull().all():
            continue

        try:
            # --- RETREAT ---
            rt = ts_r.where((doy_retreat >= 233) | (doy_retreat <= 60), drop=True)
            if (rt > THRESHOLD).any():
                idx_rt = find_first_event(rt, THRESHOLD, WINDOW, above=False)
                if not np.isnan(idx_rt):
                    retreat[j, i] = rt.time[idx_rt].dt.dayofyear.item()

            # --- ADVANCE ---
            ad = ts_a.where((doy_advance >= 30) & (doy_advance <= 274), drop=True)
            if (ad > THRESHOLD).any():
                idx_ad = find_first_event(ad, THRESHOLD, WINDOW, above=True)
                if not np.isnan(idx_ad):
                    advance[j, i] = ad.time[idx_ad].dt.dayofyear.item()

        except Exception:
            continue

# === EXPORT === #
out_ds = xr.Dataset(
    {
        f"advance_{YEAR}": (("y", "x"), advance),
        f"retreat_{YEAR}": (("y", "x"), retreat),
    },
    coords={"x": ds.x, "y": ds.y},
    attrs={
        "description": f"SMMR Advance & Retreat | THRESHOLD={THRESHOLD}, WINDOW={WINDOW}, Year={YEAR}"
    }
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_file = os.path.join(OUTPUT_DIR, f"seaice_phases_SMMR_{YEAR}_test.nc")
out_ds.to_netcdf(out_file)
print(f"✅ Saved {out_file}")

del advance, retreat, out_ds, data_retreat, data_advance
gc.collect()
