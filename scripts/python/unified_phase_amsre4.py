# unified_phase_AMSRE.py

import os
import numpy as np
import xarray as xr
from pathlib import Path

# === CONFIG ===

# LOCAL:
# INPUT_FILE = "/Users/fridaperez/Developer/repos/sea-ice-phase/data/amsre/merged_amsre_SH_2012_2024.nc"
# OUTPUT_DIR = "/Users/fridaperez/Developer/repos/sea-ice-phase/results/AMSRE_phase/"

# CLUSTER:
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/amsre/merged_amsre_SH_2012_2024.nc"
OUTPUT_DIR = "/user/geog/falejandraperez/sea-ice-phase/results/AMSRE_phase/"

THRESHOLD = 15  # Note: AMSRE is in % (not 0–1)
WINDOW = 5

# === FUNCTIONS ===

def continuous_meet(cond, window_size, dim='time'):
    _found = cond.rolling({dim: window_size}, center=True).sum(skipna=True).fillna(0)
    detected = _found >= window_size
    indices = (detected * np.arange(detected.shape[0]).reshape(-1, 1, 1)).max(axis=0)
    return xr.DataArray(indices, coords={k: v for k, v in cond.coords.items() if k != dim},
                        dims=[d for d in cond.dims if d != dim])

# === LOAD DATA ===

ds = xr.open_dataset(INPUT_FILE)
sic = ds[[v for v in ds.data_vars if "ICECON" in v][0]]
years = np.unique(sic.time.dt.year.values)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# === LOOP THROUGH YEARS ===

for year in years:
    print(f"Processing {year}...")

    yearly = sic.sel(time=sic.time.dt.year == year)

    doy = yearly.time.dt.dayofyear
    cond_retreat = (doy >= 213) | (doy <= 59)  # Aug–Feb
    cond_advance = (doy >= 32) & (doy <= 273)  # Feb–Sep

    retreat_sic = yearly.where(cond_retreat)
    advance_sic = yearly.where(cond_advance)

    # Retreat: find first 5-day period <= threshold
    retreat_idx = continuous_meet(retreat_sic <= THRESHOLD, WINDOW)
    # Advance: find first 5-day period >= threshold
    advance_idx = continuous_meet(advance_sic >= THRESHOLD, WINDOW)

    # Convert indices back to DOY
    retreat_doy = yearly.time[0].dt.dayofyear + retreat_idx
    advance_doy = yearly.time[0].dt.dayofyear + advance_idx

    out_ds = xr.Dataset(
        {
            "retreat": retreat_doy,
            "advance": advance_doy
        },
        coords={"x": sic.x, "y": sic.y},
        attrs={"description": f"AMSRE Phase Timing for {year}, window={WINDOW}d, threshold={THRESHOLD}%"}
    )

    out_file = os.path.join(OUTPUT_DIR, f"seaice_phases_AMSRE_{year}.nc")
    out_ds.to_netcdf(out_file)
    print(f"✅ Saved: {out_file}")
